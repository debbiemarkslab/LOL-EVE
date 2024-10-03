import json
import os
import torch
from torch import nn
import numpy as np
from argparse import ArgumentParser
from torch.optim import Adam
from transformers import PreTrainedTokenizerFast, CTRLConfig, CTRLLMHeadModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
import wandb
import random

class SequenceDataModule(LightningDataModule):
    def __init__(self, batch_size, sequence_file, tokenizer, val_split, max_positional_embedding_size, cache, use_weighted_sampler, num_embd, model_device, validation_chromosome):
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
        self.num_cpus = 2
        self.cache = cache
        self.use_weighted_sampler = use_weighted_sampler
        self.num_embd = num_embd
        self.model_device = model_device
        self.validation_chromosome = validation_chromosome
        
        
    def prepare_data(self):
        self.dataset = load_dataset("parquet", data_files=self.sequence_file, split="train", cache_dir=self.cache)

    def setup(self, stage=None):

        def tokenize_batch(batch):
            inputs = []

            batch_size = len(batch['gene'])
            for index in range(batch_size):
                sequence = 'start ' + ' '.join(batch['sequence'][index]) + ' end'
                inputs.append(f"{batch['gene'][index]} [SEP] {batch['species'][index]} [SEP] {batch['clade'][index]} [SEP] {sequence}")

            batch['input'] = inputs

            assert "input" in batch.keys(), batch
            out = self.tokenizer(batch["input"], padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_positional_embedding_size)
            out['input_ids'] = out['input_ids'].long()
            if out['input_ids'].max() >= len(self.tokenizer):
                raise ValueError("Token ID out of range")
            batch["input_ids"] = out['input_ids']
            return batch
        
        # Split data into train and validation sets
        self.train_dataset = self.dataset.filter(lambda example: example["chr"] != self.validation_chromosome)
        self.val_dataset = self.dataset.filter(lambda example: example["chr"] == self.validation_chromosome)

        self.train_dataset = self.train_dataset.remove_columns(['chr'])
        self.val_dataset = self.val_dataset.remove_columns(['chr'])

        if self.use_weighted_sampler:
            self.train_weights = self.train_dataset["normalized_score"]
            self.val_weights = self.val_dataset["normalized_score"]

        # Remove the score and normalized_score columns if they exist
        if 'score' in self.train_dataset.column_names and 'normalized_score' in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.remove_columns(['score', 'normalized_score'])
        if 'score' in self.val_dataset.column_names and 'normalized_score' in self.val_dataset.column_names:
            self.val_dataset = self.val_dataset.remove_columns(['score', 'normalized_score'])

        if 'embedding' in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.remove_columns(['embedding'])

        if 'embedding' in self.val_dataset.column_names:
            self.val_dataset = self.val_dataset.remove_columns(['embedding'])

        print(f"Train dataset length: {len(self.train_dataset)}")
        print(f"Validation dataset length: {len(self.val_dataset)}")

        self.train_dataset.set_transform(tokenize_batch)
        self.val_dataset.set_transform(tokenize_batch)

        self.vocab_size = len(self.tokenizer)  # Store vocab size after setup

    def collate_fn(self, batch):
        # Ensure that each element in the batch is a tensor
        input_ids = torch.stack([torch.tensor(item['input_ids']) if not isinstance(item['input_ids'], torch.Tensor) else item['input_ids'] for item in batch])
        return {'input_ids': input_ids}

    def train_dataloader(self):
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights))
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0, pin_memory=True, collate_fn=self.collate_fn)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True, collate_fn=self.collate_fn)


class InferenceSequenceDatasetLLR(Dataset):
    def __init__(self, wt_sequences, variant_sequences, gene_tokens, species_tokens, clade_tokens, tokenizer, max_length, use_control_codes=True):
        self.wt_sequences = wt_sequences
        self.variant_sequences = variant_sequences
        self.gene_tokens = gene_tokens
        self.species_tokens = species_tokens
        self.clade_tokens = clade_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_control_codes = use_control_codes

    def __len__(self):
        return len(self.wt_sequences)

    def __getitem__(self, idx):
        wt_sequence = self.wt_sequences[idx]
        variant_sequence = self.variant_sequences[idx]

        # Handle empty sequences
        if not wt_sequence or wt_sequence.isspace():
            wt_sequence = 'start end'
        else:
            wt_sequence = 'start ' + ' '.join(wt_sequence.lower()) + ' end'

        if not variant_sequence or variant_sequence.isspace():
            variant_sequence = 'start end'
        else:
            variant_sequence = 'start ' + ' '.join(variant_sequence.lower()) + ' end'

        
        if self.use_control_codes:
            gene_token = self.gene_tokens[idx]
            species_token = self.species_tokens[idx]
            clade_token = self.clade_tokens[idx]
            wt_combined_sequence = f"{gene_token} [SEP] {species_token} [SEP] {clade_token} [SEP] {wt_sequence}"
            variant_combined_sequence = f"{gene_token} [SEP] {species_token} [SEP] {clade_token} [SEP] {variant_sequence}"
        else:
            wt_combined_sequence = f"[SEP] [SEP] [SEP] [SEP] [SEP] [SEP] {wt_sequence}"
            variant_combined_sequence = f"[SEP] [SEP] [SEP] [SEP] [SEP] [SEP] {variant_sequence}"

        wt_tokenized = self.tokenizer(wt_combined_sequence, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length)
        variant_tokenized = self.tokenizer(variant_combined_sequence, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length)

        return {
            'wt_input_ids': wt_tokenized['input_ids'].squeeze(0).long(),
            'variant_input_ids': variant_tokenized['input_ids'].squeeze(0).long()
        }

class InferenceSequenceDatasetPerplexity(Dataset):
    def __init__(self, sequences, gene_tokens, species_tokens, clade_tokens, tokenizer, max_length, use_control_codes=True):
        self.sequences = sequences
        self.gene_tokens = gene_tokens
        self.species_tokens = species_tokens
        self.clade_tokens = clade_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_control_codes = use_control_codes
        self.cache = None
        self.use_weighted_sampler = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # here u can decide to randomly truncate
        sequence ='start ' + ' '.join(sequence.lower()) +' end'
        if self.use_control_codes:
            gene_token = self.gene_tokens[idx]
            species_token = self.species_tokens[idx]
            clade_token = self.clade_tokens[idx]
            combined_sequence = f"{gene_token} [SEP] {species_token} [SEP] {clade_token} [SEP] {sequence}"
        else:
            combined_sequence = f"[SEP] [SEP] [SEP] [SEP] [SEP] [SEP] {sequence}"

        tokenized = self.tokenizer(combined_sequence, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length)
        input_ids = tokenized['input_ids'].squeeze(0).long()  # Remove the batch dimension
        return {'input_ids': input_ids}


class InferenceSequenceDataModuleLLR(SequenceDataModule):
    def set_sequences(self, wt_sequences, variant_sequences, gene_tokens, species_tokens, clade_tokens, use_control_codes=True):
        self.dataset = InferenceSequenceDatasetLLR(wt_sequences, variant_sequences, gene_tokens, species_tokens, clade_tokens, self.tokenizer, self.max_positional_embedding_size, use_control_codes)

    def setup(self, stage=None):
        pass

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_cpus, pin_memory=True)


class InferenceSequenceDataModulePerplexity(SequenceDataModule):
    def set_sequences(self, sequences, gene_tokens, species_tokens, clade_tokens, use_control_codes=True):
        self.dataset = InferenceSequenceDatasetPerplexity(sequences, gene_tokens, species_tokens, clade_tokens, self.tokenizer, self.max_positional_embedding_size, use_control_codes)

    def setup(self, stage=None):
        pass

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_cpus, pin_memory=True, shuffle=False)


class PromEVEModel(LightningModule):
    def __init__(self, tokenizer, num_layers, num_embd, num_heads, max_positional_embedding_size, lr, embeddings_file, model_device):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_embd = num_embd
        self.num_heads = num_heads
        self.max_positional_embedding_size = max_positional_embedding_size
        self.lr = lr
        self.esm_embedding = np.load(embeddings_file)
        self.embedding_genes = self.esm_embedding.files
        self.tokenizer_genes = list(set(self.tokenizer.vocab.keys()) & set(self.embedding_genes))
        self.vocab_size = None  # Initialized later
        self.model = None  # Model initialization deferred
        self.gene_reduction = nn.Linear(1280, num_embd, bias=False)  # Adjusted input size to 1280, output size to 768, no bias
        self.training_step_outputs = []  # Initialize here
        self.validation_step_outputs = []  # Initialize here
        self.model_device = model_device
        self.sample_embedding = torch.nn.Embedding(len(self.tokenizer.added_tokens_decoder), num_embd)
        self.gene_embedding = torch.nn.Embedding(len(self.tokenizer_genes) + 1, self.esm_embedding['samd11'].shape[0])

        #initialize gene embedding
        for index, gene in enumerate(self.tokenizer_genes):
            self.gene_embedding.weight.data[index] = torch.from_numpy(self.esm_embedding[gene])

        for param in self.gene_embedding.parameters():
            param.requires_grad = False

        # make gene_token to embedding dict
        # map from token to gene_string
        # gene_string to get the embedding
        
        self.token_to_gene_embedding_index = {}
        for index, gene in enumerate(self.tokenizer_genes):
            gene_token = tokenizer.convert_tokens_to_ids(gene)
            self.token_to_gene_embedding_index[gene_token] = index

        # accounts for separation token
        empty_vector = torch.empty_like(torch.tensor(self.esm_embedding['samd11']))
        empty_vector.requires_grad_(True)
        self.gene_embedding.weight.data[(len(self.tokenizer_genes))] = empty_vector
        self.token_to_gene_embedding_index[2] = len(self.tokenizer_genes)
            
        # still need to make gene_embedding be the same size as sample_embedding

    def prepare_model(self, vocab_size):
        self.vocab_size = vocab_size
        model_config = CTRLConfig.from_pretrained("ctrl", vocab_size=vocab_size, n_layer=self.num_layers, n_embd=self.num_embd, n_head=self.num_heads, n_positions=self.max_positional_embedding_size, output_attentions=True)
        self.model = CTRLLMHeadModel(model_config)
        self.model.to(self.model_device)  # Move model to the correct device
    
    def forward(self, input_ids):
        assert input_ids.dtype == torch.long
        # so here i need to update the gene_token to be the appropriate gene_embedding
        # need to manually make embeddings for each token
        gene_tokens = input_ids[:,1]

        # dictionary
        #gene_index = self.token_to_embedding[gene_tokens]

        # Convert gene tokens to embedding indexes
        embedding_indexes = torch.tensor([self.token_to_gene_embedding_index.get(token.item(), self.token_to_gene_embedding_index[2]) 
                                        for token in gene_tokens], 
                                        dtype=torch.long,  # Ensure long (integer) dtype
                                        device=self.model_device)
        embedding_indexes = embedding_indexes.to(self.model_device) 
        current_gene_embeddings = self.gene_embedding(embedding_indexes)
        current_gene_embedding = self.gene_reduction(current_gene_embeddings)

        # Get the first element in the second dimension
        cls_token = input_ids[:, 0]
        remaining_tokens = input_ids[:, 2:]

        cls_embedding = self.sample_embedding(cls_token)
        remaining_embeddings = self.sample_embedding(remaining_tokens)
        
        #batch_size = input_ids.shape[0]
        # inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Reshape cls_embedding and current_gene_embedding to have the same number of dimensions as remaining_embeddings
        cls_embedding = cls_embedding.unsqueeze(1)  # Shape: (1, 1, 768)
        current_gene_embedding = current_gene_embedding.unsqueeze(1)  # Shape: (1, 1, 768)
        inputs_embeds = torch.cat((cls_embedding, current_gene_embedding, remaining_embeddings ), dim=1)  # Shape: (1, 1010, 768)


        #inputs_embeds = torch.cat((current_gene_embedding, remaining_embeddings), dim=1)
        
        return self.model(inputs_embeds=inputs_embeds, labels=input_ids)

    def calculate_loss(self, input_ids, logger_loss, logger_perplexity):

        # add random sampling of control tokens
        combinations = torch.tensor([(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]])
        positions_to_potentially_change = [1, 3, 5]
        sep_token = 2

        # Randomly select a combination of positions to mask for each entry in the batch
        batch_size = input_ids.size(0)
        random_indices = torch.randint(0, len(combinations), (batch_size,))
        random_combinations = combinations[random_indices]

        # Create a mask for the positions to change
        mask = random_combinations == 0

        # Create a tensor to apply the SEP token
        sep_tensor = torch.full_like(input_ids, sep_token)

        sep_tensor = sep_tensor.to(self.model_device)
        mask = mask.to(self.model_device)

        # Apply the SEP token to the selected positions
        for i, pos in enumerate(positions_to_potentially_change):
            input_ids[:, pos] = torch.where(mask[:, i], sep_tensor[:, pos], input_ids[:, pos])

        outputs = self.forward(input_ids)
        logits = outputs.logits[:,:-1]
        labels = input_ids.clone()

        # Identify the start and end tokens
        start_token = self.tokenizer.convert_tokens_to_ids('start')
        end_token = self.tokenizer.convert_tokens_to_ids('end')

        # Find the positions of the start and end tokens, include the token for the loss
        start_position = (labels == start_token).nonzero(as_tuple=False)[0][1].item() + 1
        end_position = (labels == end_token).nonzero(as_tuple=False)[0][1].item()

        # Apply masks
        labels[:,:start_position] = 3
        labels[:,end_position:] = 3

        labels = labels[:, 1:]

        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean', ignore_index=3)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='mean', ignore_index=3)
        perplexity = torch.exp(loss)
        cpu_loss = loss.clone().detach().cpu().item()
        self.log(logger_loss, loss.item(), sync_dist=True, on_step=True)
        self.log(logger_perplexity, perplexity, sync_dist=True, on_step=True)

        return loss, cpu_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        loss, cpu_loss = self.calculate_loss(input_ids, 'train_loss', 'train_perplexity')
        self.training_step_outputs.append(cpu_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        loss, cpu_loss = self.calculate_loss(input_ids, 'val_loss', 'val_perplexity')
        self.validation_step_outputs.append(cpu_loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([torch.tensor(x) for x in self.training_step_outputs]).mean()
        train_perplexity = torch.exp(avg_train_loss)
        self.log('train_loss', avg_train_loss.item(), sync_dist=True)
        self.log('train_perplexity', train_perplexity.item(), sync_dist=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([torch.tensor(x) for x in self.validation_step_outputs]).mean()
        val_perplexity = torch.exp(avg_val_loss)
        self.log('val_loss', avg_val_loss.item(), sync_dist=True)
        self.log('val_perplexity', val_perplexity.item(), sync_dist=True)
        self.validation_step_outputs.clear()
    
