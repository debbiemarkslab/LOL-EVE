import torch
from torch import nn
import numpy as np
from datasets import load_dataset
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import Adam
from transformers import CTRLConfig, CTRLLMHeadModel
import pandas as pd



class SequenceDataModule(LightningDataModule):
    def __init__(self, batch_size, sequence_file, tokenizer, val_split, max_positional_embedding_size, 
                 cache, use_weighted_sampler, num_embd, model_device, validation_chromosome, num_cpus):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_file = sequence_file
        self.val_split = val_split
        self.tokenizer = tokenizer
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.max_positional_embedding_size = max_positional_embedding_size
        self.vocab_size = len(self.tokenizer)
        self.num_cpus = num_cpus
        self.cache = cache
        self.use_weighted_sampler = use_weighted_sampler
        self.num_embd = num_embd
        self.model_device = model_device
        self.validation_chromosome = validation_chromosome
        self.has_directions = False

    def prepare_data(self):
        self.dataset = load_dataset("parquet", data_files=self.sequence_file, split="train", cache_dir=self.cache)

    def setup(self, stage=None):
        def tokenize_batch(batch):
            inputs = []
            batch_size = len(batch['gene'])
            
            for index in range(batch_size):

                # Clean and prepare identifiers - ensure they're treated as single tokens
                gene = batch['gene'][index].strip()
                species = batch['species'][index].strip()
                clade = batch['clade'][index].strip()

                # Get the sequence and split into tokens
                sequence_tokens = batch['sequence'][index].strip()
                # If sequence is too long, trim it
                # We have room for 1000 sequence tokens
                if len(sequence_tokens) > 1000:
                    sequence_tokens = sequence_tokens[:1000]

                assert (len(sequence_tokens) + 7) <= self.max_positional_embedding_size   
                # Ensure we're adding exactly one start and one end token
                
                sequence = ' '.join(sequence_tokens)
                
                input_text = f"{gene} {species} {clade} [SOS] {sequence} [EOS]"
                inputs.append(input_text)

            batch['input'] = inputs
            assert "input" in batch.keys(), batch
            
            out = self.tokenizer(
                batch["input"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_positional_embedding_size
            )
            out['input_ids'] = out['input_ids'].long()
            if out['input_ids'].max() >= len(self.tokenizer):
                raise ValueError("Token ID out of range")
            if self.has_directions:
                out['direction'] = batch['direction']
            # Get token IDs for start and end tokens
            start_token_id = self.tokenizer.convert_tokens_to_ids('[SOS]')
            end_token_id = self.tokenizer.convert_tokens_to_ids('[EOS]')

            # Validate each sequence in the batch
            for idx, ids in enumerate(out['input_ids']):
                # Count occurrences of start and end tokens
                start_count = (ids == start_token_id).sum().item()
                end_count = (ids == end_token_id).sum().item()
                
                # Check for exactly one start and one end token
                assert start_count == 1, \
                    f"Found {start_count} start tokens in sequence {idx}, should be exactly 1. Input text: {inputs[idx]}"
                assert end_count == 1, \
                    f"Found {end_count} end tokens in sequence {idx}, should be exactly 1. Input text: {inputs[idx]}"
                
                # Get the single positions
                start_pos = (ids == start_token_id).nonzero().item()
                end_pos = (ids == end_token_id).nonzero().item()
                
                # Check sequence length
                assert len(ids) == self.max_positional_embedding_size, \
                    f"Expected {self.max_positional_embedding_size} tokens, got {len(ids)}"
                
                # Check token order
                assert start_pos < end_pos, \
                    f"Start token must come before end token in sequence {idx}, input text: {inputs[idx]}, {ids}, {start_pos}, {end_pos}"

            batch["input_ids"] = out['input_ids']
            
            return batch

        # Split data into train and validation sets
        if not self.validation_chromosome:
            self.train_dataset = self.dataset
            self.val_dataset = None  # No validation dataset        
        else:
            self.train_dataset = self.dataset.filter(lambda example: example["chrom"] != self.validation_chromosome)
            self.val_dataset = self.dataset.filter(lambda example: example["chrom"] == self.validation_chromosome)

        self.train_dataset = self.train_dataset.remove_columns(['chrom'])
        #self.val_dataset = self.val_dataset.remove_columns(['chrom'])
        
        # Only process validation dataset if it exists
        if self.val_dataset is not None:
            self.val_dataset = self.val_dataset.remove_columns(['chrom'])

        if self.use_weighted_sampler:
            self.train_weights = self.train_dataset["normalized_score"]
            if self.val_dataset is not None:
                self.val_weights = self.val_dataset["normalized_score"]
        
        columns_to_remove = ['score', 'normalized_score', 'embedding']

        if 'direction' in self.train_dataset.column_names:
            self.has_directions = True
        else:
            columns_to_remove.append('direction')            

        for col in columns_to_remove:
            if col in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.remove_columns([col])
            if (self.val_dataset is not None) and (col in self.val_dataset.column_names):
                self.val_dataset = self.val_dataset.remove_columns([col])

        # Set transforms
        self.train_dataset.set_transform(tokenize_batch)
        if self.val_dataset is not None:
            self.val_dataset.set_transform(tokenize_batch)
    
        print(f"Train dataset length: {len(self.train_dataset)}")
        if self.val_dataset is not None:
            print(f"Validation dataset length: {len(self.val_dataset)}")
        else:
            print("No validation dataset (training on full dataset)")

    def collate_fn(self, batch):
        input_ids = torch.stack([
            torch.tensor(item['input_ids']) if not isinstance(item['input_ids'], torch.Tensor)
            else item['input_ids'] for item in batch
        ])
        
        output = {'input_ids': input_ids}
        
        if self.has_directions and 'direction' in batch[0]:
            directions = torch.tensor([
                1 if item['direction'] == 'towards' else -1 for item in batch
            ], dtype=torch.long)
            
            output['direction'] = directions
            
        return output

    def train_dataloader(self):
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights))
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                sampler=sampler, 
                num_workers=self.num_cpus, 
                pin_memory=True, 
                collate_fn=self.collate_fn
            )
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_cpus, 
            pin_memory=True, 
            collate_fn=self.collate_fn
        )
    def val_dataloader(self):
        if self.val_dataset is None:
            return None  # Return None if no validation dataset
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_cpus, 
            pin_memory=True, 
            collate_fn=self.collate_fn
        )   


class LOLEVE(LightningModule):
    def __init__(self, tokenizer, num_layers, num_embd, num_heads, max_positional_embedding_size, 
                 lr, weight_decay, embeddings_file, model_device, gpu_settings, use_control_codes=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_embd = num_embd
        self.num_heads = num_heads
        self.max_positional_embedding_size = max_positional_embedding_size
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.lr = gpu_settings['initial_lr']
        self.esm_embedding = np.load(embeddings_file)
        self.embedding_genes = self.esm_embedding.files
        self.tokenizer_genes = list(set(self.tokenizer.vocab.keys()) & set(self.embedding_genes))
        self.vocab_size = None
        self.model = None
        self.gene_reduction = nn.Linear(1280, num_embd, bias=False)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.model_device = model_device
        self.gpu_settings = gpu_settings
        self.use_control_codes = use_control_codes
        self.sample_embedding = nn.Embedding(len(self.tokenizer.added_tokens_decoder), num_embd)
        self.gene_embedding = nn.Embedding(len(self.tokenizer_genes) + 1, self.esm_embedding['samd11'].shape[0])
        
        nn.init.xavier_uniform_(self.sample_embedding.weight, gain=0.5)
        
        self.special_token_map = {
            'mask': self.tokenizer.convert_tokens_to_ids('[MASK]'),
            'start': self.tokenizer.convert_tokens_to_ids('[SOS]'),
            'end': self.tokenizer.convert_tokens_to_ids('[EOS]')
        }
        
        self.control_positions = {
            'gene': 1,
            'species': 2,
            'clade': 3
        }
        
        self.sequence_start = 4
        
        self._initialize_gene_embeddings()

    def _initialize_gene_embeddings(self):
            """Initialize gene embeddings with careful initialization."""
            for index, gene in enumerate(self.tokenizer_genes):
                self.gene_embedding.weight.data[index] = torch.from_numpy(self.esm_embedding[gene])
            
            for param in self.gene_embedding.parameters():
                param.requires_grad = False
                
            self.token_to_gene_embedding_index = {}
            for index, gene in enumerate(self.tokenizer_genes):
                gene_token = self.tokenizer.convert_tokens_to_ids(gene)
                self.token_to_gene_embedding_index[gene_token] = index
                
            # Create empty vector for mask token
            empty_vector = torch.empty_like(torch.tensor(self.esm_embedding['samd11']))
            # Use standard normal initialization instead of xavier_uniform for 1D tensor
            torch.nn.init.normal_(empty_vector, mean=0.0, std=0.02)
            
            self.gene_embedding.weight.data[len(self.tokenizer_genes)] = empty_vector
            self.token_to_gene_embedding_index[self.special_token_map['mask']] = len(self.tokenizer_genes)

    def prepare_model(self, vocab_size):
        self.vocab_size = vocab_size
        model_config = CTRLConfig.from_pretrained(
            "ctrl",
            vocab_size=vocab_size,
            n_layer=self.num_layers,
            n_embd=self.num_embd,
            n_head=self.num_heads,
            n_positions=self.max_positional_embedding_size,
            output_attentions=True,
            use_cache=True,
            initializer_range=0.02
        )
        
        self.model = CTRLLMHeadModel(model_config)
        
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
                
        if self.gpu_settings['compile_model']:
            torch.cuda.empty_cache()
            self.model = torch.compile(self.model)
            
        self.model.to(self.model_device)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        
        total_steps = self.trainer.max_steps
        warmup_steps = int(total_steps * 0.05)  # 5% warmup
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.lr * 0.1
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }    


    def forward(self, input_ids, position_ids=None):
        assert input_ids.dtype == torch.long
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        gene_tokens = input_ids[:, self.control_positions['gene']]
        embedding_indexes = torch.tensor([
            self.token_to_gene_embedding_index.get(
                token.item(), 
                self.token_to_gene_embedding_index[self.special_token_map['mask']]
            ) 
            for token in gene_tokens
        ], dtype=torch.long, device=self.model_device)
        
        current_gene_embeddings = self.gene_embedding(embedding_indexes)
        current_gene_embedding = self.gene_reduction(current_gene_embeddings)
        
        control_tokens = input_ids[:, :self.sequence_start]
        sequence_tokens = input_ids[:, self.sequence_start:]
        
        control_embeddings = self.sample_embedding(control_tokens)
        sequence_embeddings = self.sample_embedding(sequence_tokens)
        
        control_embeddings[:, self.control_positions['gene']] = current_gene_embedding
        
        inputs_embeds = torch.cat((control_embeddings, sequence_embeddings), dim=1)
        
        return self.model(
            inputs_embeds=inputs_embeds,
            labels=input_ids,
            position_ids=position_ids
        )

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
                sos_token = self.special_token_map['start']
                eos_token = self.special_token_map['end']
                
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
                ]).squeeze()

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
        start_token = self.special_token_map['start']
        end_token = self.special_token_map['end']
        
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

        loss, metrics = self.calculate_loss(input_ids, 'train', 'train_perplexity', ablation_config=None, directions=directions)
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
            loss, metrics = self.calculate_loss(
                input_ids.clone(),
                f'val_{config["name"]}',
                f'val_perplexity_{config["name"]}',
                config['config']
            )
            results[config['name']] = metrics
        
        self.validation_step_outputs.append(results)
        return results['no_ablation']['loss']
    

    def on_validation_epoch_end(self):
        """Handle end of validation epoch, computing averages and logging metrics."""
        from collections import defaultdict
        
        aggregated = {}
        
        for outputs in self.validation_step_outputs:
            for config_name, metrics in outputs.items():
                if config_name not in aggregated:
                    aggregated[config_name] = defaultdict(list)
                for metric_name, value in metrics.items():
                    # Ensure values are floats
                    aggregated[config_name][metric_name].append(float(value))
        
        for config_name, metrics in aggregated.items():
            for metric_name, values in metrics.items():
                # Convert to tensor with float dtype
                avg_value = torch.tensor(values, dtype=torch.float32).mean().item()
                self.log(
                    f'val_{config_name}_{metric_name}_avg',
                    avg_value,
                    sync_dist=True
                )
        
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        """Handle end of training epoch, computing averages and logging metrics."""
        from collections import defaultdict
        
        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)
        
        for metrics in self.training_step_outputs:
            for name, value in metrics.items():
                metrics_sum[name] += float(value)
                metrics_count[name] += 1
        
        for name in metrics_sum:
            avg_value = metrics_sum[name] / metrics_count[name]
            self.log(f'train_{name}_epoch', avg_value, sync_dist=True)
        
        self.training_step_outputs.clear()

class InferenceSequenceDataset(Dataset):
    """Dataset for variant scoring during inference."""
    def __init__(self, wt_sequences, variant_sequences, gene_tokens, species_tokens, 
                 clade_tokens, tokenizer, max_length):
        self.wt_sequences = wt_sequences
        self.variant_sequences = variant_sequences
        self.gene_tokens = gene_tokens
        self.species_tokens = species_tokens
        self.clade_tokens = clade_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.wt_sequences)

    def __getitem__(self, idx):
        wt_sequence = self.wt_sequences[idx]
        variant_sequence = self.variant_sequences[idx]
        
        gene_token = '[UNK]' if self.gene_tokens[idx] not in self.tokenizer.vocab else self.gene_tokens[idx]
        gene_token = gene_token.strip() if pd.notna(gene_token) else '[MASK]'
        #gene_token = self.gene_tokens[idx].strip() if pd.notna(self.gene_tokens[idx]) else '[MASK]'
        species_token = self.species_tokens[idx].strip() if pd.notna(self.species_tokens[idx]) else '[MASK]'
        clade_token = self.clade_tokens[idx].strip() if pd.notna(self.clade_tokens[idx]) else '[MASK]'
        
        wt_combined = f"{gene_token} {species_token} {clade_token} [SOS] {wt_sequence} [EOS]"
        variant_combined = f"{gene_token} {species_token} {clade_token} [SOS] {variant_sequence} [EOS]"

        wt_tokenized = self.tokenizer(
            wt_combined, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        variant_tokenized = self.tokenizer(
            variant_combined, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length
	    )

        for tokenized in [wt_tokenized, variant_tokenized]:
            input_ids = tokenized['input_ids'].squeeze(0)
            start_count = (input_ids == self.tokenizer.convert_tokens_to_ids('[SOS]')).sum().item()
            end_count = (input_ids == self.tokenizer.convert_tokens_to_ids('[EOS]')).sum().item()
   
            assert start_count == 1 and end_count == 1, f"Invalid token counts in sequence {idx}, {start_count}, {end_count}, {input_ids[0:20]}, {input_ids[-20:]}"
            assert len(input_ids) == self.max_length, f"Length mismatch in sequence {idx}"

        return {
            'wt_input_ids': wt_tokenized['input_ids'].squeeze(0).long(),
            'variant_input_ids': variant_tokenized['input_ids'].squeeze(0).long()
        }

class InferenceSequenceDataModuleLLR(SequenceDataModule):
    """Data module for Log Likelihood Ratio computation during inference."""
    def set_sequences(self, wt_sequences, variant_sequences, gene_tokens, species_tokens, 
                     clade_tokens):
        self.dataset = InferenceSequenceDataset(
            wt_sequences, variant_sequences, gene_tokens, species_tokens, clade_tokens,
            self.tokenizer, self.max_positional_embedding_size
        )

    def setup(self, stage=None):
        pass

    def collate_fn(self, batch):
        wt_ids = torch.stack([
            torch.tensor(item['wt_input_ids']) if not isinstance(item['wt_input_ids'], torch.Tensor)
            else item['wt_input_ids'] for item in batch
        ])
        var_ids = torch.stack([
            torch.tensor(item['variant_input_ids']) if not isinstance(item['variant_input_ids'], torch.Tensor)
            else item['variant_input_ids'] for item in batch
        ])
        return {
            'wt_input_ids': wt_ids,
            'variant_input_ids': var_ids
        }
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_cpus, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )
