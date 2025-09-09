import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
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

class SequenceDataModule(LightningDataModule):
    """Unified data module for DNA sequence tasks"""
    
    def __init__(self, 
                batch_size,
                tokenizer,
                val_split=0.1,
                max_positional_embedding_size=1007,
                sequence_file=None,
                cache=None,
                use_weighted_sampler=False,
                validation_chromosome=None,
                num_cpus=4,
                model_device=None):
        """
        Args:
            batch_size: Batch size for dataloaders
            tokenizer: Huggingface tokenizer
            val_split: Validation split ratio (if not using chromosome split)
            max_positional_embedding_size: Maximum sequence length
            sequence_file: Path to sequence parquet file
            cache: Cache directory for datasets
            use_weighted_sampler: Whether to use weighted sampling
            validation_chromosome: Chromosome to use for validation
            num_cpus: Number of CPUs for dataloaders
            model_device: Device to place the model on
        """
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.val_split = val_split
        self.max_positional_embedding_size = max_positional_embedding_size
        self.sequence_file = sequence_file
        self.cache = cache
        self.use_weighted_sampler = use_weighted_sampler
        self.validation_chromosome = validation_chromosome
        self.num_cpus = num_cpus
        self.model_device = model_device if model_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize datasets as None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Set vocab size
        self.vocab_size = len(self.tokenizer)
        
        # Track whether dataset has direction information
        self.has_directions = False

    def prepare_data(self):
        """Load the dataset from file"""
        if self.sequence_file:
            self.dataset = load_dataset(
                "parquet", 
                data_files=self.sequence_file, 
                split="train", 
                cache_dir=self.cache
            )
            logger.info(f"Loaded dataset with {len(self.dataset)} sequences")

    def tokenize_batch(self, batch):
        """Tokenize a batch of sequences with control codes"""
        inputs = []
        batch_size = len(batch['gene'])
        
        for index in range(batch_size):
            # Clean and prepare identifiers
            gene = batch['gene'][index].strip()
            species = batch['species'][index].strip()
            clade = batch['clade'][index].strip()

            # Get sequence tokens
            sequence_tokens = batch['sequence'][index].strip()
            
            # Trim if too long
            if len(sequence_tokens) > 1000:
                sequence_tokens = sequence_tokens[:1000]

            # Ensure sequence fits within max length
            assert (len(sequence_tokens) + 7) <= self.max_positional_embedding_size   
            
            # Format sequence with proper spacing
            sequence = ' '.join(sequence_tokens)
            
            # Create full input with control codes and sequence tokens
            input_text = f"{gene} {species} {clade} [SOS] {sequence} [EOS]"
            inputs.append(input_text)

        # Add inputs to batch
        batch['input'] = inputs
        
        # Tokenize batch
        tokenized = self.tokenizer(
            batch["input"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_positional_embedding_size
        )
        
        # Convert to long tensor
        tokenized['input_ids'] = tokenized['input_ids'].long()
        
        # Check for out-of-range token IDs
        if tokenized['input_ids'].max() >= len(self.tokenizer):
            raise ValueError("Token ID out of range")
        
        # Add directions if present
        if self.has_directions:
            tokenized['direction'] = batch['direction']
        
        # Get token IDs for start and end tokens
        start_token_id = self.tokenizer.convert_tokens_to_ids('[SOS]')
        end_token_id = self.tokenizer.convert_tokens_to_ids('[EOS]')

        # Validate each sequence in the batch
        for idx, ids in enumerate(tokenized['input_ids']):
            # Count occurrences of start and end tokens
            start_count = (ids == start_token_id).sum().item()
            end_count = (ids == end_token_id).sum().item()
            
            # Check for exactly one start and one end token
            assert start_count == 1, \
                f"Found {start_count} start tokens in sequence {idx}, should be exactly 1. Input text: {inputs[idx]}"
            assert end_count == 1, \
                f"Found {end_count} end tokens in sequence {idx}, should be exactly 1. Input text: {inputs[idx]}"
            
            # Get the positions
            start_pos = (ids == start_token_id).nonzero().item()
            end_pos = (ids == end_token_id).nonzero().item()
            
            # Check sequence length
            assert len(ids) == self.max_positional_embedding_size, \
                f"Expected {self.max_positional_embedding_size} tokens, got {len(ids)}"
            
            # Check token order
            assert start_pos < end_pos, \
                f"Start token must come before end token in sequence {idx}, input text: {inputs[idx]}, {ids}, {start_pos}, {end_pos}"

        # Add token IDs to batch
        batch["input_ids"] = tokenized['input_ids']
        
        return batch
    
    def setup(self, stage=None):
        """Setup datasets for training, validation, or inference"""
        # Always prepare data if not already done
        if not self.dataset and self.sequence_file:
            self.prepare_data()
        
        # If using training data from file
        if self.dataset:
            if stage == 'fit' or stage is None:
                self._setup_for_training()
            
            elif stage == 'test' or stage == 'validate':
                self._setup_for_validation()
    
    def _setup_for_training(self):
        """Setup training and validation datasets"""
        # Determine chromosome column name
        chrom_col = 'chr' if 'chr' in self.dataset.column_names else 'chrom'
        
        # Split by chromosome or randomly
        if self.validation_chromosome and chrom_col in self.dataset.column_names:
            logger.info(f"Using chromosome {self.validation_chromosome} for validation")
            train_data = self.dataset.filter(
                lambda example: example[chrom_col] != self.validation_chromosome
            )
            val_data = self.dataset.filter(
                lambda example: example[chrom_col] == self.validation_chromosome
            )
            
            # Remove chromosome column
            train_data = train_data.remove_columns([chrom_col])
            val_data = val_data.remove_columns([chrom_col])
        else:
            # Use random split
            logger.info(f"Using random {self.val_split:.1%} split for validation")
            splits = self.dataset.train_test_split(test_size=self.val_split)
            train_data = splits['train']
            val_data = splits['test']
        
        # Check for direction information
        if 'direction' in train_data.column_names:
            self.has_directions = True
            logger.info("Dataset contains sequence direction information")
        
        # Remove unnecessary columns
        columns_to_remove = ['score', 'normalized_score', 'embedding']
        for col in columns_to_remove:
            if col in train_data.column_names:
                train_data = train_data.remove_columns([col])
            if col in val_data.column_names:
                val_data = val_data.remove_columns([col])
        
        # Setup weighted sampling if requested
        if self.use_weighted_sampler and 'normalized_score' in train_data.column_names:
            self.train_weights = train_data["normalized_score"]
            self.val_weights = val_data["normalized_score"]
            logger.info("Using weighted sampling based on normalized_score")
        
        # Set transform functions
        train_data.set_transform(self.tokenize_batch)
        val_data.set_transform(self.tokenize_batch)
        
        # Save the processed datasets
        self.train_dataset = train_data
        self.val_dataset = val_data
        
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def _setup_for_validation(self):
        """Setup for validation only"""
        # Use validation chromosome if specified
        if self.validation_chromosome:
            chrom_col = 'chr' if 'chr' in self.dataset.column_names else 'chrom'
            val_data = self.dataset.filter(
                lambda example: example[chrom_col] == self.validation_chromosome
            )
            val_data = val_data.remove_columns([chrom_col])
            
            # Set transform function
            val_data.set_transform(self.tokenize_batch)
            
            # Save the processed dataset
            self.val_dataset = val_data
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def setup_for_variant_scoring(self, wt_sequences, var_sequences, control_codes):
        """Setup for variant scoring"""
        # Create dataset for variant scoring
        self.val_dataset = SequenceDataset(
            wt_sequences, 
            var_sequences, 
            control_codes['gene'], 
            control_codes['species'], 
            control_codes['clade'],
            self.tokenizer, 
            self.max_positional_embedding_size
        )
        logger.info(f"Setup variant scoring dataset with {len(wt_sequences)} variant pairs")
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        if 'input_ids' in batch[0]:
            # Training/validation mode
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
        else:
            # Variant scoring mode
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
    
    def train_dataloader(self):
        """Create training dataloader"""
        if self.train_dataset is None:
            return None
            
        if self.use_weighted_sampler and hasattr(self, 'train_weights'):
            sampler = WeightedRandomSampler(
                self.train_weights, 
                len(self.train_weights)
            )
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                sampler=sampler, 
                num_workers=self.num_cpus, 
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        else:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_cpus, 
                pin_memory=True,
                collate_fn=self.collate_fn
            )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_cpus, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        """Create test dataloader (same as validation)"""
        return self.val_dataloader()
    
    @staticmethod
    def from_variants_file(variants_path, tokenizer, batch_size=16, max_length=1007):
        """Create a data module from a variants CSV file
        
        Args:
            variants_path: Path to variants CSV
            tokenizer: Tokenizer to use
            batch_size: Batch size for dataloader
            max_length: Maximum sequence length
            
        Returns:
            Configured SequenceDataModule
        """
        # Load variants
        variants_df = pd.read_csv(variants_path)
        
        # Extract sequences and metadata
        wt_sequences = variants_df['WT'].tolist()
        var_sequences = variants_df['VAR'].tolist()
        
        # Process control codes
        control_codes = {
            'gene': variants_df['GENE'].str.lower().tolist(),
            'species': variants_df['SPECIES'].str.lower().tolist(),
            'clade': variants_df['CLADE'].str.lower().tolist()
        }
        
        # Create data module
        data_module = SequenceDataModule(
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_positional_embedding_size=max_length
        )
        
        # Setup for variant scoring
        data_module.setup_for_variant_scoring(
            wt_sequences,
            var_sequences,
            control_codes
        )
        
        return data_module