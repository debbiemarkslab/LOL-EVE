import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from transformers import CTRLConfig, CTRLLMHeadModel
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from tqdm import tqdm
from .scoring import calculate_sequence_loss, create_reverse_complement  # Add this import
from .embeddings import AdaptiveLocalPositionEmbedding, RoPEPositionEmbedding
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

class LOLEVE(LightningModule):
    """
    LOLEVE model for genomic sequence modeling
    and variant effect prediction. This model combines pre-trained protein embeddings
    with a causal language modeling approach for DNA sequences.
    """
    
    def __init__(self, 
                 tokenizer,
                 num_layers=12, 
                 num_embd=768, 
                 num_heads=12, 
                 max_positional_embedding_size=1007,
                 lr=1e-5,
                 weight_decay=0.0,
                 embeddings_file=None,
                 use_control_codes=1,
                 model_device=None,
                 gpu_settings=None,
                 position_embedding_type='rope'):
        """
        Initialize the LOLEVE model
        
        Args:
            tokenizer: Huggingface tokenizer
            num_layers: Number of transformer layers
            num_embd: Dimension of embeddings
            num_heads: Number of attention heads
            max_positional_embedding_size: Maximum sequence length
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            embeddings_file: Path to pre-trained gene embeddings file (NPZ)
            model_device: Device to place the model on
            gpu_settings: Dictionary of GPU-specific settings
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_embd = num_embd
        self.num_heads = num_heads
        self.max_positional_embedding_size = max_positional_embedding_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_device = model_device if model_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_settings = gpu_settings if gpu_settings else {
            'precision': 'bf16-mixed',
            'compile_model': False,
            'initial_lr': lr,
            'gradient_clip_val': 1.0
        }
        
        # Load gene embeddings
        self.esm_embedding = np.load(embeddings_file)
        self.embedding_genes = self.esm_embedding.files
        self.tokenizer_genes = list(set(self.tokenizer.vocab.keys()) & set(self.embedding_genes))
        
        # Initialize model components
        self.vocab_size = None  # Will be set in prepare_model
        self.model = None       # Will be initialized in prepare_model
        self.gene_reduction = nn.Linear(1280, num_embd, bias=False)  # Reduce ESM embedding dimension
        
        # For tracking metrics during training
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Token embeddings for non-gene tokens
        self.sample_embedding = nn.Embedding(len(self.tokenizer.added_tokens_decoder), num_embd)
        
        # Gene embedding layer
        self.gene_embedding = nn.Embedding(len(self.tokenizer_genes) + 1, self.esm_embedding['samd11'].shape[0])
        
        # Initialize gene embeddings
        self._initialize_gene_embeddings()
        
        # Define sequence positions
        self.sequence_start = 4  # Position after control tokens
        self.control_positions = {
            'gene': 1,
            'species': 2,
            'clade': 3
        }
        
        # Get token IDs for special tokens
        self.special_token_ids = self._get_special_token_ids()
        self.use_control_codes = use_control_codes
        self.position_embedding_type = position_embedding_type
        self.special_token_map = self.special_token_ids  # For backward compatibility


        if position_embedding_type == 'adaptive':
            self.position_embedding = AdaptiveLocalPositionEmbedding(
                d_model=num_embd,
                sequence_start=self.sequence_start,
                max_len=max_positional_embedding_size
            )
        elif position_embedding_type == 'rope':
            self.position_embedding = RoPEPositionEmbedding(
                d_model=num_embd,
                sequence_start=self.sequence_start,
                max_len=max_positional_embedding_size
            )
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding_type}")
            
    
    def _get_special_token_ids(self):
        """Get IDs for special tokens"""
        return {
            'sos': self.tokenizer.convert_tokens_to_ids('[SOS]'),
            'eos': self.tokenizer.convert_tokens_to_ids('[EOS]'),
            'pad': self.tokenizer.pad_token_id,
            'mask': self.tokenizer.convert_tokens_to_ids('[MASK]'),
            'sep': self.tokenizer.convert_tokens_to_ids('[SEP]')
        }
    
    def _initialize_gene_embeddings(self):
        """Initialize gene embeddings with pre-trained values"""
        for index, gene in enumerate(self.tokenizer_genes):
            self.gene_embedding.weight.data[index] = torch.from_numpy(self.esm_embedding[gene])

        # Freeze gene embeddings - they come pre-trained
        for param in self.gene_embedding.parameters():
            param.requires_grad = False
        
        # Create mapping from token IDs to embedding indices
        self.token_to_gene_embedding_index = {}
        for index, gene in enumerate(self.tokenizer_genes):
            gene_token = self.tokenizer.convert_tokens_to_ids(gene)
            self.token_to_gene_embedding_index[gene_token] = index

        # Handle separator token with an empty embedding vector
        empty_vector = torch.empty_like(torch.tensor(self.esm_embedding['samd11']))
        torch.nn.init.normal_(empty_vector, mean=0.0, std=0.02)
        self.gene_embedding.weight.data[len(self.tokenizer_genes)] = empty_vector
        self.token_to_gene_embedding_index[self.tokenizer.convert_tokens_to_ids('[SEP]')] = len(self.tokenizer_genes)
        
        # Initialize sample embedding with Xavier initialization
        nn.init.xavier_uniform_(self.sample_embedding.weight, gain=0.5)
    
    def prepare_model(self, vocab_size):
        """
        Initialize the transformer model
        
        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        
        # Create CTRL configuration
        model_config = CTRLConfig.from_pretrained(
            "ctrl",
            vocab_size=vocab_size,
            n_layer=self.num_layers,
            n_embd=self.num_embd,
            n_head=self.num_heads,
            n_positions=self.max_positional_embedding_size,
            output_attentions=True,
            use_cache=True
        )
        
        # Initialize model
        self.model = CTRLLMHeadModel(model_config)
        
        # Apply Xavier initialization to all linear layers
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
        
        # Compile model if enabled
        if self.gpu_settings.get('compile_model', False):
            torch.cuda.empty_cache()
            self.model = torch.compile(self.model)
        
        # Move model to the appropriate device
        self.model.to(self.model_device)
    
    def forward(self, input_ids, position_ids=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of token indices (batch_size, seq_len)
            position_ids: Optional tensor of position indices
            
        Returns:
            Model outputs (logits, loss, etc.)
        """
        assert input_ids.dtype == torch.long
        
        # Use provided position IDs or create default ones
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Extract gene tokens
        gene_tokens = input_ids[:, self.control_positions['gene']]
        
        # Map gene tokens to embedding indices
        embedding_indexes = torch.tensor([
            self.token_to_gene_embedding_index.get(
                token.item(), 
                self.token_to_gene_embedding_index[self.tokenizer.convert_tokens_to_ids('[SEP]')]
            ) 
            for token in gene_tokens
        ], dtype=torch.long, device=self.model_device)
        
        # Get gene embeddings and reduce dimension
        current_gene_embeddings = self.gene_embedding(embedding_indexes)
        current_gene_embedding = self.gene_reduction(current_gene_embeddings.detach())
        
        # Split control tokens and sequence tokens
        control_tokens = input_ids[:, :self.sequence_start]
        sequence_tokens = input_ids[:, self.sequence_start:]
        
        # Get embeddings for tokens
        control_embeddings = self.sample_embedding(control_tokens)
        sequence_embeddings = self.sample_embedding(sequence_tokens)
        
        # Replace gene token embedding with the gene-specific one
        control_embeddings[:, self.control_positions['gene']] = current_gene_embedding
        
        # Combine control and sequence embeddings
        inputs_embeds = torch.cat((control_embeddings, sequence_embeddings), dim=1)

        # Apply adaptive local position embeddings (new part)
        inputs_embeds = self.position_embedding(inputs_embeds, input_ids, self.tokenizer)
        batch_size, seq_length, _ = inputs_embeds.size()
        dummy_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=inputs_embeds.device)

        # # Forward pass through transformer
        return self.model(
             inputs_embeds=inputs_embeds,
             labels=input_ids,
             position_ids=dummy_position_ids
         )
        #return self.model(
        #    inputs_embeds=inputs_embeds,
        #    labels=input_ids,
        #    position_ids=position_ids
        #)

    def calculate_loss(self, input_ids, logger_loss, logger_perplexity, ablation_config=None, directions=None):
        """Calculate loss ignoring padded tokens but using full sequence."""
        mask_token = self.special_token_map['mask']
        pad_token = self.tokenizer.pad_token_id
        batch_size = len(input_ids)
        ablated_ids = input_ids.clone()
        valid_sequence_mask = torch.ones_like(ablated_ids, dtype=torch.bool)
        
        # Training mode:
        if logger_loss == 'train':

            # control token dropout
            pos_indices = torch.tensor(list(self.control_positions.values())).long()
            embedding_length = ablated_ids.shape[-1] 
            all_indices = torch.arange(0, embedding_length)
            mask = torch.rand(ablated_ids.shape) > 0.9
            no_mask = torch.isin(all_indices, pos_indices).unsqueeze(0).repeat(batch_size, 1)
            ablated_ids[no_mask * mask] = mask_token

            # length dropout
            if directions is not None:
                # Find [SOS] and [EOS] positions
                sos_token = self.special_token_map['sos']
                eos_token = self.special_token_map['eos']
                
                # Get positions for each sequence in batch
                sos_positions = (ablated_ids == sos_token).nonzero(as_tuple=True)[1]  # [batch_size]
                eos_positions = (ablated_ids == eos_token).nonzero(as_tuple=True)[1]  # [batch_size]
                
                # Calculate sequence lengths and dropout lengths
                seq_lengths = eos_positions - sos_positions - 1
                max_dropouts = (seq_lengths * 0.9).long()
                dropout_lengths = torch.stack([
                    torch.randint(0, max_dropout + 1, (1,), device=ablated_ids.device) 
                    if max_dropout > 0 else torch.tensor([0], device=ablated_ids.device)
                    for max_dropout in max_dropouts
                ]).squeeze(1)

                # Handle forward direction (shift left + pad right)
                forward_indices = torch.where(directions == 1)[0]
                if len(forward_indices) > 0:
                    # Create temporary buffer for just the forward sequences
                    temp_buffer = ablated_ids[forward_indices].clone()
                    
                    for i, idx in enumerate(forward_indices):
                        seq_start = sos_positions[idx] + 1
                        seq_end = eos_positions[idx]
                        dropout_len = dropout_lengths[idx]
                        
                        # Create a temporary sequence for the shift
                        sequence = temp_buffer[i, seq_start+dropout_len:seq_end].clone()
                        # Fill in the shifted sequence
                        temp_buffer[i, seq_start:seq_end-dropout_len] = sequence
                        # Add padding at the end
                        temp_buffer[i, seq_end-dropout_len:seq_end] = pad_token
                    
                    # Update only the forward sequences
                    ablated_ids[forward_indices] = temp_buffer

                # Handle reverse direction (just pad right)
                reverse_indices = torch.where(directions == -1)[0]
                if len(reverse_indices) > 0:
                    for idx in reverse_indices:
                        seq_end = eos_positions[idx]
                        dropout_len = dropout_lengths[idx]
                        # Replace end of sequence with padding
                        ablated_ids[idx, seq_end-dropout_len:seq_end] = pad_token

                # Add assertions to verify sequence integrity
                # Verify total sequence lengths haven't changed

                for i in range(batch_size):
                    original_length = eos_positions[i] - sos_positions[i] - 1
                    current_length = (ablated_ids[i, sos_positions[i]+1:eos_positions[i]] != pad_token).sum()
                    non_pad_length = current_length.item()
                    assert non_pad_length + dropout_lengths[i] == original_length, \
                        f"Sequence length mismatch for sample {i}: Original {original_length}, Current non-pad {non_pad_length}, Dropout {dropout_lengths[i]}"
                    
                    # Verify SOS and EOS tokens are still in place
                    assert ablated_ids[i, sos_positions[i]].item() == sos_token, \
                        f"SOS token missing or moved for sample {i}"
                    assert ablated_ids[i, eos_positions[i]].item() == eos_token, \
                        f"EOS token missing or moved for sample {i}"
                    
                    # Verify no pad tokens before sequence starts
                    assert not (ablated_ids[i, :sos_positions[i]] == pad_token).any(), \
                        f"Found pad tokens before SOS in sample {i}"
                    
                    # Verify direction-specific padding
                    if directions[i] == 1:
                        # For forward direction, verify padding is at the end
                        pad_section = ablated_ids[i, eos_positions[i]-dropout_lengths[i]:eos_positions[i]]
                        assert (pad_section == pad_token).all(), \
                            f"Forward direction sample {i} doesn't have padding at the end"
                    else:
                        # For reverse direction, verify padding is at the end
                        pad_section = ablated_ids[i, eos_positions[i]-dropout_lengths[i]:eos_positions[i]]
                        assert (pad_section == pad_token).all(), \
                            f"Reverse direction sample {i} doesn't have padding at the end"



            # Validation mode
        else:
            ablation_config = ablation_config or {
                'gene': False, 
                'species': False, 
                'clade': False
            }
            
            for control_type, should_mask in ablation_config.items():
                if should_mask and control_type in self.control_positions:
                    pos = self.control_positions[control_type]
                    ablated_ids[:, pos] = mask_token
       
        # vanilla loleve hack here
        if not self.use_control_codes:
            #print('no control tags')
            for pos in self.control_positions.values():
                ablated_ids[:, pos] = mask_token 
        
        # Forward pass with full sequence including padding
        outputs = self.forward(ablated_ids)
        logits = outputs.logits[:, :-1]  # Remove last position

        # Verify shapes match expected dimensions
        batch_size, seq_length, vocab_size = logits.shape
        assert batch_size == input_ids.size(0), f"Batch size mismatch: {batch_size} vs {input_ids.size(0)}"
        
        # Prepare labels and attention mask
        labels = ablated_ids.clone()

        # Identify the start and end tokens
        start_token = self.special_token_map['sos']
        end_token = self.special_token_map['eos']
        
        start_positions = (labels == start_token).nonzero(as_tuple=False)
        end_positions = (labels == end_token).nonzero(as_tuple=False)

        # Create position tensors
        batch_size, seq_length = labels.shape
        position_indices = torch.arange(seq_length, device=labels.device).expand(batch_size, -1)

        # Create masks for before start and after end positions
        start_mask = position_indices >= (start_positions[:, 1] + 1).unsqueeze(1)
        end_mask = position_indices < end_positions[:, 1].unsqueeze(1)
        pad_mask = (labels != pad_token)        

        # Combine masks and apply
        valid_positions = start_mask & end_mask & pad_mask
        labels = torch.where(valid_positions, labels, torch.tensor(-100, device=labels.device))

        # Remove first token from all sequences in batch
        labels = labels[:, 1:]
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none',
            ignore_index=-100
        )
        
        # Reshape loss and apply mask
        loss = loss.reshape(logits.size(0), -1)
        valid_positions = (labels != -100).float()

        # this 
        sequence_loss = ((loss * valid_positions).sum(dim=1) / (valid_positions.sum(dim=1))).mean()
        sequence_loss = sequence_loss.requires_grad_(True)  # Add this line
        
        # Handle any NaN or Inf values
        if not torch.isfinite(sequence_loss).all():
            self.log(f'{logger_loss}_nan_inf_detected', 1.0, sync_dist=True)
            sequence_loss = torch.where(
                torch.isfinite(sequence_loss), 
                sequence_loss, 
                torch.tensor(0.0, device=sequence_loss.device)
            )
            self.zero_grad()
        
        # Calculate perplexity
        perplexity = torch.exp(torch.clamp(sequence_loss, max=20))
        
        # Calculate accuracy on non-padded sequence tokens
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct_predictions = (predictions == labels) & (labels != -100)
            sequence_accuracy = correct_predictions.sum().float() / (valid_positions.sum())
        
        # Collect metrics
        metrics = {
            'loss': float(sequence_loss.detach().cpu().item()),
            'perplexity': float(perplexity.detach().cpu().item()),
            'accuracy': float(sequence_accuracy.detach().cpu().item()),
            'masked_tokens': float((ablated_ids == mask_token).sum().item()),
            'valid_tokens': float(valid_positions.sum().item())
        }
        
        # Log metrics
        for name, value in metrics.items():
            self.log(
                f'{logger_loss}_{name}', 
                value, 
                sync_dist=True, 
                on_step=True,
            )
        
        return sequence_loss, metrics

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        
        directions = batch.get('direction', None)
        if directions is not None:
            directions = directions.to(self.device)
        
        # Debug info
        if self.global_step % 10 == 0:
            sample_idx = 0
            sample_seq = input_ids[sample_idx].cpu().tolist()
            print(f"Input tokens (sample): {sample_seq[:10]}...")
            
            # Count token types
            token_counts = {}
            for tok in sample_seq:
                if tok not in token_counts:
                    token_counts[tok] = 0
                token_counts[tok] += 1
            
            print(f"Token distribution: {token_counts}")
        
        # Use the original parameter names
        loss, metrics = self.calculate_loss(
            input_ids, 
            'train',  # logger_loss parameter 
            'train_perplexity',  # logger_perplexity parameter
            ablation_config=None, 
            directions=directions
        )
        
        # Make sure loss requires grad
        if not loss.requires_grad:
            print("WARNING: Loss doesn't require grad!")
            loss = loss.clone().requires_grad_(True)
        
        # Print loss info
        if self.global_step % 10 == 0:
            print(f"Step {self.global_step} - Loss: {loss.item():.4f}, Perplexity: {metrics['perplexity']:.4f}")
        
        self.training_step_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        
        ablation_configs = [
            {'name': 'no_ablation', 'config': {'gene': False, 'species': False, 'clade': False}},
            {'name': 'gene_only', 'config': {'gene': True, 'species': False, 'clade': False}},
            {'name': 'species_only', 'config': {'gene': False, 'species': True, 'clade': False}},
            {'name': 'clade_only', 'config': {'gene': False, 'species': False, 'clade': True}},
            {'name': 'all_control', 'config': {'gene': True, 'species': True, 'clade': True}},
        ]
        
        results = {}
        for config in ablation_configs:
            # Use the original parameter names
            loss, metrics = self.calculate_loss(
                input_ids.clone(),
                f'val_{config["name"]}',  # logger_loss
                f'val_perplexity_{config["name"]}',  # logger_perplexity
                config['config']  # ablation_config
            )
            results[config['name']] = metrics
        
        self.validation_step_outputs.append(results)
        return results['no_ablation']['loss']

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler"""
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),  # Better values for transformer training
            eps=1e-8
        )
        
        # Calculate total steps and warmup
        if self.trainer is not None:
            total_steps = self.trainer.max_steps
            warmup_steps = min(2000, int(total_steps * 0.1))  # 10% warmup, max 2000 steps
            
            # Create warmup scheduler
            warmup_scheduler = LambdaLR(
                optimizer,
                lambda step: min(1.0, float(step) / float(max(1, warmup_steps)))
            )
            
            # Create cosine annealing scheduler for after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.lr * 0.1
            )
            
            # Combine the two schedulers
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        
        # Fallback if trainer not available
        return optimizer

    def on_train_end(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def on_train_epoch_end(self):
        """Process metrics at the end of training epoch"""
        if not self.training_step_outputs:
            return
            
        metrics_sum = {key: 0.0 for key in self.training_step_outputs[0].keys()}
        metrics_count = len(self.training_step_outputs)
        
        for metrics in self.training_step_outputs:
            for name, value in metrics.items():
                metrics_sum[name] += value
        
        for name in metrics_sum:
            avg_value = metrics_sum[name] / metrics_count
            self.log(f'train_{name}_epoch', avg_value, sync_dist=True)
        
        self.training_step_outputs.clear()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Handle benchmark evaluation after training batch"""
        # Call parent method
        super().on_train_batch_end(outputs, batch, batch_idx)
        
        # Check if we have benchmark attributes
        if hasattr(self, "benchmark_logger") and hasattr(self, "benchmark_freq"):
            # Check if it's time to evaluate benchmarks
            if self.global_step % self.benchmark_freq == 0:
                # Run benchmark evaluation
                self.benchmark_logger.evaluate_and_log(self.global_step)
    
    def on_validation_epoch_end(self):
        """Process metrics at the end of validation epoch"""
        if not self.validation_step_outputs:
            return
                
        # Initialize aggregated metrics
        aggregated = {}
        
        # Collect metrics from all validation steps
        for outputs in self.validation_step_outputs:
            for config_name, metrics in outputs.items():
                if config_name not in aggregated:
                    aggregated[config_name] = {key: [] for key in metrics.keys()}
                
                for metric_name, value in metrics.items():
                    # Handle tensors by converting to scalar
                    if torch.is_tensor(value):
                        if value.numel() > 1:
                            value = value.mean().item()
                        else:
                            value = value.item()
                    
                    # Add scalar value to the list
                    aggregated[config_name][metric_name].append(value)
                
        # Calculate averages and log metrics
        for config_name, metrics in aggregated.items():
            for metric_name, values in metrics.items():
                # Ensure we have a non-empty list of values
                if values:
                    # Calculate the mean of the values
                    avg_value = sum(values) / len(values)
                    self.log(
                        f'val_{config_name}_{metric_name}_epoch',
                        avg_value,
                        sync_dist=True
                    )
        
        self.validation_step_outputs.clear()
        
    def create_reverse_complement(self, input_ids):
        """Create reverse complement of input sequences preserving control tokens"""
        device = input_ids.device
        batch_size = input_ids.size(0)
        reverse_ids = input_ids.clone()
        
        # Create complement map
        complement_map = {
            self.tokenizer.vocab['a']: self.tokenizer.vocab['t'],
            self.tokenizer.vocab['t']: self.tokenizer.vocab['a'],
            self.tokenizer.vocab['c']: self.tokenizer.vocab['g'],
            self.tokenizer.vocab['g']: self.tokenizer.vocab['c']
        }
        
        sos_token = self.special_token_ids['sos']
        eos_token = self.special_token_ids['eos']
        
        for batch_idx in range(batch_size):
            # Find sequence positions
            sequence_part = input_ids[batch_idx, self.sequence_start:]
            sos_pos = (sequence_part == sos_token).nonzero(as_tuple=True)[0]
            
            if len(sos_pos) == 0:
                continue
                
            sos_pos = sos_pos[0].item() + self.sequence_start
            
            # Find end position
            eos_pos = (input_ids[batch_idx, sos_pos:] == eos_token).nonzero(as_tuple=True)[0]
            
            if len(eos_pos) == 0:
                continue
                
            eos_pos = eos_pos[0].item() + sos_pos
            
            # Extract sequence (excluding SOS/EOS tokens)
            seq = input_ids[batch_idx, sos_pos+1:eos_pos]
            
            # Create complemented and reversed sequence
            try:
                complemented = torch.tensor([
                    complement_map.get(n.item(), n.item()) for n in seq
                ], device=device)
                
                reversed_comp = torch.flip(complemented, dims=[0])
                
                # Place back in tensor
                reverse_ids[batch_idx, sos_pos+1:sos_pos+1+len(reversed_comp)] = reversed_comp
            except Exception as e:
                print(f"Error in reverse complement: {e}")
                continue
        
        return reverse_ids

    def score_variants(self, wt_ids, var_ids, normalize=False, ablation_config=None):
        """
        Score variants using log-likelihood ratio
        
        Args:
            wt_ids: Tensor of wild-type sequence IDs
            var_ids: Tensor of variant sequence IDs
            normalize: Whether to normalize scores by sequence length
            ablation_config: Optional dict for control code ablation
            
        Returns:
            Tuple of (avg_llr, forward_llr) tensors
        """
        # Use the shared scoring function from scoring.py
        # try:
        from .scoring import score_variants as score_variants_func
        
        # Try to use the scoring function
        avg_llr, forward_llr = score_variants_func(self, wt_ids, var_ids, normalize, ablation_config)
        return avg_llr, forward_llr
        # except Exception as e:
        #     print(f"Falling back to direct calculation: {str(e)}")
        #     # Fall back to direct calculation
        #     _, wt_metrics = self.calculate_loss(wt_ids)
        #     _, var_metrics = self.calculate_loss(var_ids)
            
        #     wt_loss = wt_metrics.get('per_sequence_loss')
        #     var_loss = var_metrics.get('per_sequence_loss')
            
        #     # Calculate LLR
        #     llr = -(var_loss - wt_loss)
            
        #     # Return both the same value for avg and forward
        #     return llr, llr
    def score_variants_batch(self, batch, normalize=False, ablation_config=None):
        """
        Score a batch of variants
        
        Args:
            batch: Dictionary with 'wt_input_ids' and 'variant_input_ids'
            normalize: Whether to normalize scores by sequence length
            ablation_config: Optional dict for control code ablation
            
        Returns:
            List of variant scores
        """
        wt_ids = batch['wt_input_ids'].to(self.device)
        var_ids = batch['variant_input_ids'].to(self.device)
        
        scores = self.score_variants(
            wt_ids, 
            var_ids, 
            normalize=normalize,
            ablation_config=ablation_config
        )
        
        return scores.cpu().tolist()
