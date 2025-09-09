#!/usr/bin/env python3
"""
Script to calculate log-likelihood of sequences using a trained LOL-EVE model.
This script handles variable-length sequences, including cases where VAR sequences
are shorter than the fixed-length WT sequences.
"""
import argparse
import logging
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("score_sequences")

def setup_model(config_path, model_checkpoint_path, device):
    """
    Setup the LOLEVE model from a checkpoint
    
    Args:
        config_path: Path to model config file
        model_checkpoint_path: Path to model checkpoint
        device: Device to load model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    
    # Extract parameters
    training_params = config['training_parameters']
    dev_params = config.get('development_parameters', {})
    
    # Load tokenizer
    tokenizer_path = training_params['tokenizer_dir']
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{tokenizer_path}/tokenizer.json",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="[SOS]",
        eos_token="[EOS]"
    )
    
    # Import locally to avoid circular imports
    from core.models import LOLEVE
    from core.utils import load_model_checkpoint
    
    # Initialize model
    position_embedding_type = training_params.get('position_embedding_type', 'rope')
    logger.info(f"Using position embedding type: {position_embedding_type}")
    
    model = LOLEVE(
        tokenizer=tokenizer,
        num_layers=training_params['num_layers'],
        num_embd=training_params['num_embd'],
        num_heads=training_params['n_head'],
        max_positional_embedding_size=training_params['max_positional_embedding_size'],
        lr=training_params['lr'],
        weight_decay=training_params.get('weight_decay', 0.0),
        embeddings_file=training_params['embeddings_file'],
        model_device=device,
        use_control_codes=dev_params.get('use_control_codes', 1),
        position_embedding_type=position_embedding_type
    )
    
    # Prepare model
    model.prepare_model(len(tokenizer))
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {model_checkpoint_path}")
    load_model_checkpoint(model, model_checkpoint_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def calculate_sequence_loss(model, input_ids, normalize=True):
    """
    Calculate log-likelihood (negative loss) for a batch of sequences
    
    Args:
        model: LOLEVE model
        input_ids: Tensor of input token IDs
        normalize: Whether to normalize by sequence length
        
    Returns:
        Tensor of log-likelihoods (negative losses)
    """
    from core.scoring import calculate_sequence_loss as model_calc_loss
    
    with torch.no_grad():
        # Use the model's calculate_sequence_loss function
        _, metrics = model_calc_loss(model, input_ids)
        
        # Extract per-sequence loss
        per_sequence_loss = metrics.get('per_sequence_loss', 
                                      torch.tensor([float('nan')] * input_ids.size(0), 
                                                 device=input_ids.device))
        
        # If normalizing by sequence length
        if normalize:
            # Count non-padding tokens
            pad_token = model.special_token_ids['pad']
            seq_lengths = (input_ids != pad_token).sum(dim=1).float()
            # Avoid division by zero
            seq_lengths = torch.clamp(seq_lengths, min=1.0)
            # Normalize loss by sequence length
            per_sequence_loss = per_sequence_loss / seq_lengths
        
        # Convert loss to log-likelihood (negative loss)
        log_likelihood = -per_sequence_loss
        
        return log_likelihood

def score_sequences(model, tokenizer, sequences_df, output_path, batch_size=16, normalize=True, 
                   score_both=True, ablation_config=None):
    """
    Calculate log-likelihood for sequences using the LOLEVE model
    
    Args:
        model: LOLEVE model
        tokenizer: Tokenizer
        sequences_df: DataFrame with sequences to score
        output_path: Path for output CSV
        batch_size: Batch size for inference
        normalize: Whether to normalize scores by sequence length
        score_both: Whether to score both WT and VAR sequences
        ablation_config: Optional dict for control code ablation
    
    Returns:
        DataFrame with scores
    """
    from core.data import SequenceDataModule
    
    # Verify required columns
    required_columns = ['GENE', 'SPECIES', 'CLADE', 'WT']
    missing_columns = [col for col in required_columns if col not in sequences_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check if we're scoring variants or just sequences
    score_variants = 'VAR' in sequences_df.columns and score_both
    
    if score_variants:
        # Prepare sequences for scoring both WT and VAR
        print('Score Both!')
        wt_seqs = sequences_df['WT'].tolist()
        var_seqs = sequences_df['VAR'].tolist()
        
        # Reorganize control codes
        control_codes = {
            'gene': sequences_df['GENE'].str.lower().tolist(),
            'species': sequences_df['SPECIES'].str.lower().tolist(),
            'clade': sequences_df['CLADE'].str.lower().tolist()
        }
        
        # Create data module
        logger.info(f"Creating data module for {len(sequences_df)} sequence pairs")
        data_module = SequenceDataModule(
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_positional_embedding_size=model.max_positional_embedding_size
        )
        
        # Setup for variant scoring
        data_module.setup_for_variant_scoring(wt_seqs, var_seqs, control_codes)
        
        # Get dataloader
        dataloader = data_module.val_dataloader()
        
        # Score sequences
        wt_scores = []
        var_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Scoring"):
                try:
                    # Get both WT and VAR sequences
                    wt_ids = batch['wt_input_ids'].to(model.device)
                    var_ids = batch['variant_input_ids'].to(model.device)
                    
                    # Apply ablation if requested
                    if ablation_config:
                        from core.scoring import prepare_for_scoring
                        wt_ids = prepare_for_scoring(model, wt_ids, ablation_config)
                        var_ids = prepare_for_scoring(model, var_ids, ablation_config)
                    
                    # Calculate log-likelihood for both
                    wt_ll = calculate_sequence_loss(model, wt_ids, normalize=normalize)
                    var_ll = calculate_sequence_loss(model, var_ids, normalize=normalize)
                    
                    # Store scores
                    wt_scores.extend(wt_ll.cpu().tolist())
                    var_scores.extend(var_ll.cpu().tolist())
                    
                except Exception as e:
                    # Log the error
                    logger.error(f"Error processing batch: {str(e)}")
                    # Add NaN values for failed sequences
                    batch_size = len(batch['wt_input_ids'])
                    wt_scores.extend([float('nan')] * batch_size)
                    var_scores.extend([float('nan')] * batch_size)
        
        # Add scores to DataFrame
        sequences_df['wt_log_likelihood'] = wt_scores
        sequences_df['var_log_likelihood'] = var_scores
        
        # Calculate log-likelihood ratio
        sequences_df['llr'] = sequences_df['var_log_likelihood'] - sequences_df['wt_log_likelihood']
        
        # Calculate perplexity
        sequences_df['wt_perplexity'] = np.exp(-np.array(wt_scores))
        sequences_df['var_perplexity'] = np.exp(-np.array(var_scores))
        
    else:
        # We're only scoring WT sequences
        seqs = sequences_df['WT'].tolist()
        
        # Use WT sequences for both WT and VAR (data module requires both)
        dummy_seqs = seqs.copy()
        
        # Reorganize control codes
        control_codes = {
            'gene': sequences_df['GENE'].str.lower().tolist(),
            'species': sequences_df['SPECIES'].str.lower().tolist(),
            'clade': sequences_df['CLADE'].str.lower().tolist()
        }
        
        # Create data module
        logger.info(f"Creating data module for {len(sequences_df)} sequences")
        data_module = SequenceDataModule(
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_positional_embedding_size=model.max_positional_embedding_size
        )
        
        # Setup for scoring
        data_module.setup_for_variant_scoring(seqs, dummy_seqs, control_codes)
        
        # Get dataloader
        dataloader = data_module.val_dataloader()
        
        # Score sequences
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Scoring"):
                try:
                    # Get WT sequences
                    input_ids = batch['wt_input_ids'].to(model.device)
                    
                    # Apply ablation if requested
                    if ablation_config:
                        from core.scoring import prepare_for_scoring
                        input_ids = prepare_for_scoring(model, input_ids, ablation_config)
                    
                    # Calculate log-likelihood
                    log_likelihood = calculate_sequence_loss(model, input_ids, normalize=normalize)
                    
                    # Store scores
                    scores.extend(log_likelihood.cpu().tolist())
                    
                except Exception as e:
                    # Log the error
                    logger.error(f"Error processing batch: {str(e)}")
                    # Add NaN values for failed sequences
                    batch_size = len(batch['wt_input_ids'])
                    scores.extend([float('nan')] * batch_size)
        
        # Add scores to DataFrame
        sequences_df['log_likelihood'] = scores
        
        # Calculate perplexity
        sequences_df['perplexity'] = np.exp(-np.array(scores))
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    sequences_df.to_csv(output_path, index=False)
    
    return sequences_df

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Calculate log-likelihood of sequences with LOLEVE model")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequences', type=str, required=True, help='Path to sequences CSV')
    parser.add_argument('--output', type=str, required=True, help='Path for output CSV')
    parser.add_argument('--normalize', action='store_true', help='Normalize scores by sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--ablation', choices=['none', 'gene', 'species', 'clade', 'all'], 
                       default='none', help='Control code ablation mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer = setup_model(args.config, args.checkpoint, device)
    
    # Load sequences
    logger.info(f"Loading sequences from {args.sequences}")
    sequences_df = pd.read_csv(args.sequences)
    logger.info(f"Loaded {len(sequences_df)} sequences")
    
    # Create ablation config
    ablation_config = None
    if args.ablation != 'none':
        ablation_config = {
            'gene': args.ablation in ['gene', 'all'],
            'species': args.ablation in ['species', 'all'],
            'clade': args.ablation in ['clade', 'all']
        }
        logger.info(f"Using ablation config: {ablation_config}")
    
    # Score sequences
    results_df = score_sequences(
        model,
        tokenizer,
        sequences_df,
        args.output,
        batch_size=args.batch_size,
        normalize=args.normalize,
        score_both=True,
        ablation_config=ablation_config
    )
    
    logger.info("Scoring complete!")

if __name__ == "__main__":
    main()