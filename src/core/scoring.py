"""
Shared scoring functionality for LOL-EVE.
"""
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def create_reverse_complement(model, input_ids):
    """Create reverse complement of sequences while preserving control tokens"""
    device = input_ids.device
    batch_size = input_ids.size(0)
    reverse_ids = input_ids.clone()
    sequence_start = model.sequence_start
    
    # Get special token IDs
    special_tokens = model.special_token_ids
    sos_token = special_tokens['sos']
    eos_token = special_tokens['eos']
    
    # Create a safer complement map based on what's in the tokenizer
    complement_map = {}
    
    # Try different case combinations for nucleotides
    for base, comp in [('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C'),
                       ('a', 't'), ('t', 'a'), ('c', 'g'), ('g', 'c')]:
        try:
            # Only add to map if both base and complement exist in tokenizer
            if base in model.tokenizer.vocab and comp in model.tokenizer.vocab:
                base_id = model.tokenizer.vocab[base]
                comp_id = model.tokenizer.vocab[comp]
                complement_map[base_id] = comp_id
        except:
            # Skip if any error occurs
            continue
    
    # If no valid complement pairs were found, return the original tokens
    if not complement_map:
        return input_ids
    
    for batch_idx in range(batch_size):
        try:
            # Find sequence positions
            sequence_part = input_ids[batch_idx, sequence_start:]
            sos_pos = (sequence_part == sos_token).nonzero(as_tuple=True)[0]
            
            if len(sos_pos) == 0:
                continue
                
            sos_pos = sos_pos[0].item() + sequence_start
            
            # Find end position
            eos_pos = (input_ids[batch_idx, sos_pos:] == eos_token).nonzero(as_tuple=True)[0]
            
            if len(eos_pos) == 0:
                continue
                
            eos_pos = eos_pos[0].item() + sos_pos
            
            # Extract sequence (excluding SOS/EOS tokens)
            if sos_pos + 1 >= eos_pos or sos_pos + 1 >= input_ids.size(1) or eos_pos >= input_ids.size(1):
                continue  # Skip if indices are invalid
                
            seq = input_ids[batch_idx, sos_pos+1:eos_pos]
            
            # Create complemented and reversed sequence
            try:
                complemented = []
                for n in seq:
                    n_item = n.item()
                    # Use complement if available, otherwise keep original token
                    complemented.append(complement_map.get(n_item, n_item))
                
                complemented = torch.tensor(complemented, device=device)
                reversed_comp = torch.flip(complemented, dims=[0])
                
                # Check that the destination has enough space
                if sos_pos + 1 + len(reversed_comp) > input_ids.size(1):
                    # Truncate to fit
                    reversed_comp = reversed_comp[:input_ids.size(1) - (sos_pos + 1)]
                
                # Place back in tensor
                reverse_ids[batch_idx, sos_pos+1:sos_pos+1+len(reversed_comp)] = reversed_comp
            except Exception as e:
                print(f"Error in reverse complement: {e}")
                continue
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    return reverse_ids


def calculate_sequence_loss(model, input_ids, ignore_index=-100):
    """
    Modified sequence loss calculation with nucleotide probability normalization
    
    Args:
        model: LOL-EVE model instance
        input_ids: Tensor of token indices
        ignore_index: Value to use for ignored positions
        
    Returns:
        Tuple of (sequence_loss, metrics_dict)
    """
    # Get special token IDs
    special_tokens = model.special_token_ids
    pad_token = special_tokens['pad']
    sos_token = special_tokens['sos']
    eos_token = special_tokens['eos']
    
    # Forward pass with normalization
    outputs = model.forward(input_ids)
    
    # Get logits and normalize to nucleotide probabilities
    logits = outputs.logits[:, :-1]
    #normalized_logits = normalize_to_nucleotide_probs(model, logits)
    
    # Prepare labels
    labels = input_ids.clone()
    
    # Find sequence boundaries
    sos_positions = (labels == sos_token).nonzero(as_tuple=True)[1]
    eos_positions = (labels == eos_token).nonzero(as_tuple=True)[1]
    
    # Create position indices for masking
    batch_size, seq_length = labels.shape
    position_indices = torch.arange(seq_length, device=labels.device).expand(batch_size, -1)
    
    # Create masks for sequence positions
    start_mask = position_indices >= (sos_positions + 1).unsqueeze(1)
    end_mask = position_indices < eos_positions.unsqueeze(1)
    pad_mask = (labels != pad_token)
    
    # Combine masks to get valid positions
    valid_positions = start_mask & end_mask & pad_mask
    
    # Set ignored positions to ignore_index
    labels = torch.where(valid_positions, labels, torch.tensor(ignore_index, device=labels.device))
    
    # Remove first token for prediction
    labels = labels[:, 1:]
    
    # Calculate cross entropy loss using normalized logits
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
        ignore_index=ignore_index
    )
    
    # Reshape loss and apply mask
    loss = loss.reshape(logits.size(0), -1)
    valid_positions = (labels != ignore_index).float()
    
    # Calculate per-sequence losses
    with torch.no_grad():
        seq_losses = (loss * valid_positions).sum(dim=1) / (valid_positions.sum(dim=1) + 1e-8)
        perplexity = torch.exp(torch.clamp(seq_losses, max=20))
        predictions = logits.argmax(dim=-1)
        correct_predictions = (predictions == labels) & (labels != ignore_index)
        accuracy = correct_predictions.sum().float() / (valid_positions.sum() + 1e-8)
    
    # Calculate mean loss
    sequence_loss = seq_losses.mean()
    
    # Metrics
    metrics = {
        'loss': float(sequence_loss.detach().cpu().item()),
        'perplexity': float(perplexity.mean().detach().cpu().item()),
        'accuracy': float(accuracy.detach().cpu().item()),
        'valid_tokens': float(valid_positions.sum().item()),
        'masked_tokens': float((input_ids == special_tokens['mask']).sum().item()),
        'per_sequence_loss': seq_losses.detach()  # Return per-sequence losses for variant scoring
    }
    
    return sequence_loss, metrics

def prepare_for_scoring(model, input_ids, ablation_config=None):
    """Prepare input tokens for scoring with optional ablation
    
    Args:
        model: LOLEVE model instance
        input_ids: Tensor of token indices
        ablation_config: Optional dict for control code ablation
        
    Returns:
        Modified input IDs tensor
    """
    if not ablation_config:
        return input_ids
    
    # Create a copy to avoid modifying original
    modified_ids = input_ids.clone()
    mask_token = model.special_token_ids['mask']
    
    # Apply requested ablations
    for control_type, should_mask in ablation_config.items():
        if should_mask and control_type in model.control_positions:
            pos = model.control_positions[control_type]
            modified_ids[:, pos] = mask_token
    
    return modified_ids

def score_variants(model, wt_ids, var_ids, normalize=False, ablation_config=None):
    """Score variants using log-likelihood ratio
    
    Args:
        model: LOLEVE model instance
        wt_ids: Tensor of wild-type sequence IDs
        var_ids: Tensor of variant sequence IDs
        normalize: Whether to normalize scores by sequence length
        ablation_config: Optional dict for control code ablation
        
    Returns:
        Tuple of (avg_scores, forward_scores) for the batch
    """
    device = model.device
    
    # Apply control code ablation if requested
    if ablation_config:
        wt_ids = prepare_for_scoring(model, wt_ids, ablation_config)
        var_ids = prepare_for_scoring(model, var_ids, ablation_config)
    
    # Create reverse complement sequences
    wt_reverse = create_reverse_complement(model, wt_ids)
    var_reverse = create_reverse_complement(model, var_ids)
    
    # Calculate losses for each sequence
    with torch.no_grad():
        # Forward sequences
        _, wt_metrics = calculate_sequence_loss(model, wt_ids)
        _, var_metrics = calculate_sequence_loss(model, var_ids)
        
        # Extract per-sequence losses
        wt_loss = wt_metrics['per_sequence_loss']
        var_loss = var_metrics['per_sequence_loss']
        
        # Reverse complement sequences
        _, wt_rev_metrics = calculate_sequence_loss(model, wt_reverse)
        _, var_rev_metrics = calculate_sequence_loss(model, var_reverse)
        
        # Extract per-sequence losses
        wt_rev_loss = wt_rev_metrics['per_sequence_loss']
        var_rev_loss = var_rev_metrics['per_sequence_loss']
    
    # Average forward and reverse
    wt_avg_loss = (wt_loss + wt_rev_loss) / 2
    var_avg_loss = (var_loss + var_rev_loss) / 2
    
    # Calculate LLR
    # Negative sign: lower loss (higher probability) is better
    avg_llr = -(var_avg_loss - wt_avg_loss)
    forward_llr = -(var_loss - wt_loss)
    
    return avg_llr, forward_llr

def score_variants_batch(model, dataloader, normalize=False, ablation_config=None):
    """Score a batch of variants
    
    Args:
        model: LOLEVE model instance
        dataloader: DataLoader with variant pairs
        normalize: Whether to normalize by sequence length
        ablation_config: Optional dict for control code ablation
        
    Returns:
        List of (avg_score, forward_score) tuples
    """
    model.eval()
    device = model.device
    llr_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating LLR"):
            try:
                wt_ids = batch['wt_input_ids'].to(device)
                var_ids = batch['variant_input_ids'].to(device)
                
                # Score variants
                avg_llr, forward_llr = score_variants(
                    model, wt_ids, var_ids, normalize, ablation_config
                )
                
                # Save results as tuples
                batch_results = [
                    (avg_val.item(), fwd_val.item()) 
                    for avg_val, fwd_val in zip(avg_llr, forward_llr)
                ]
                
                llr_results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error in batch: {str(e)}")
                # Add NaN for failed sequences
                llr_results.extend([(float('nan'), float('nan'))] * len(wt_ids))
    
    return llr_results