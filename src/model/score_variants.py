import argparse
import logging
import json
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import pandas as pd
from models import (
    LOLEVE, 
    InferenceSequenceDataModuleLLR,
)
from train_lightening import detect_gpu_architecture

def calculate_llr(model, dataloader, normalize=False):
    """Calculate log-likelihood ratios using new scoring method"""
    model.eval()
    device = model.device
    
    sos_token = model.tokenizer.convert_tokens_to_ids('[SOS]')
    eos_token = model.tokenizer.convert_tokens_to_ids('[EOS]')
    pad_token = model.tokenizer.pad_token_id
    
    complement_map = {
        model.tokenizer.vocab['A']: model.tokenizer.vocab['T'],
        model.tokenizer.vocab['T']: model.tokenizer.vocab['A'],
        model.tokenizer.vocab['C']: model.tokenizer.vocab['G'],
        model.tokenizer.vocab['G']: model.tokenizer.vocab['C']
    }
    
    llr_results = []

    def compute_sequence_loss(input_ids):       
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        labels = input_ids.clone()

        start_positions = (input_ids == sos_token).nonzero(as_tuple=False)
        end_positions = (input_ids == eos_token).nonzero(as_tuple=False)
        
        batch_size, seq_length = input_ids.size()
        position_indices = torch.arange(seq_length, device=device).expand(batch_size, -1)

        start_mask = position_indices >= (start_positions[:, 1] + 1).unsqueeze(1)
        end_mask = position_indices < end_positions[:, 1].unsqueeze(1)
        pad_mask = (labels != pad_token)  

        # Combine masks and apply
        valid_positions = start_mask & end_mask & pad_mask
        labels = torch.where(valid_positions, labels, torch.tensor(-100, device=labels.device))

        labels = labels[:, 1:]

        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none',
            ignore_index=-100
        )

        loss = loss.reshape(batch_size, -1)
        valid_positions = (labels != -100).float()
       	 
        # loss = loss.reshape(batch_size, -1)
        # loss_mask = valid_positions.float()
        
        if normalize:
            #print('NORMALIZE')
            sequence_lengths = (valid_positions.sum(dim=1))
            sequence_loss = (loss * valid_positions).sum(dim=1) / sequence_lengths
            #print(f"Sequence Lengths: {sequence_lengths}")
            #print(f"Raw Loss: {(loss * valid_positions).sum(dim=1)}")
            #print(f"Normalized Loss: {sequence_loss}")
        else:
            sequence_loss = (loss * valid_positions).sum(dim=1)  
        return sequence_loss
    

    def create_reverse_complement(input_ids):
        batch_size = input_ids.size(0)
        reverse_ids = input_ids.clone()
        
        for batch_idx in range(batch_size):
            sequence_part = input_ids[batch_idx, model.sequence_start:]
            end_pos = (sequence_part == eos_token).nonzero(as_tuple=True)[0]
            
            if len(end_pos) == 0:
                continue
                
            end_pos = end_pos[0].item() + model.sequence_start
            seq = sequence_part[1:end_pos-model.sequence_start]
            
            complemented = torch.tensor([complement_map.get(n.item(), n.item()) 
                                       for n in seq], device=device)
            reversed_comp = torch.flip(complemented, dims=[0])
            
            reverse_ids[batch_idx, model.sequence_start+1:model.sequence_start+1+len(reversed_comp)] = reversed_comp
        
        return reverse_ids

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating LLR"):
            try:
                wt_ids = batch['wt_input_ids'].to(device)
                var_ids = batch['variant_input_ids'].to(device)
                
                wt_loss = compute_sequence_loss(wt_ids)
                var_loss = compute_sequence_loss(var_ids)
                
                wt_reverse = create_reverse_complement(wt_ids)
                var_reverse = create_reverse_complement(var_ids)
                
                wt_reverse_loss = compute_sequence_loss(wt_reverse)
                var_reverse_loss = compute_sequence_loss(var_reverse)
                
                wt_avg_loss = (wt_loss + wt_reverse_loss) / 2
                var_avg_loss = (var_loss + var_reverse_loss) / 2
                
                # wt - var
                # if llr + (var is path), then positive score and var must be less likely
                # if llr - (var is good), then negative score and var must be more likely
                llr = -(var_avg_loss - wt_avg_loss)
                
                # Save both averaged and forward-only LLRs
                llr_results.extend([(llr.item(), -(var_loss.item() - wt_loss.item())) 
                                  for llr, var_loss, wt_loss in zip(llr, var_loss, wt_loss)])
                
            except Exception as e:
                logging.error(f"Error in batch: {str(e)}")
                llr_results.extend([float('nan')] * len(wt_ids))
    return llr_results

def prep_variants(variants, experiment_type):
    """Extract sequences and metadata based on experiment type."""
    def create_variant_seq(row):
        base_dict = {
            'wt_seq': row['WT'].upper() if pd.notna(row['WT']) else '',
            'variant_seq': row['VAR'].upper() if pd.notna(row['VAR']) else '',
            'gene': '[MASK]',
            'species': '[MASK]',
            'clade': '[MASK]'
        }
        
        if experiment_type == 'all':
            base_dict.update({
                'gene': row['GENE'].lower() if pd.notna(row['GENE']) else '[MASK]',
                'species': row['SPECIES'].lower() if pd.notna(row['SPECIES']) else '[MASK]',
                'clade': row['CLADE'].lower() if pd.notna(row['CLADE']) else '[MASK]'
            })
        elif experiment_type == 'gene_only':
            base_dict['gene'] = row['GENE'].lower() if pd.notna(row['GENE']) else '[MASK]'
        elif experiment_type == 'species_only':
            base_dict['species'] = row['SPECIES'].lower() if pd.notna(row['SPECIES']) else '[MASK]'
        elif experiment_type == 'clade_only':
            base_dict['clade'] = row['CLADE'].lower() if pd.notna(row['CLADE']) else '[MASK]'
        
        return base_dict

    return variants.apply(create_variant_seq, axis=1).tolist()

def score_variants_ablation(config, variants_path, model_checkpoint, output_prefix, normalize=False):
    """Score variants using different control code combinations"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    training_params = config['training_parameters']
    dev_params = config['development_parameters']
    
    logger.info("Loading variants...")
    variants_df = pd.read_csv(variants_path)
    
    required_columns = ['GENE', 'SPECIES', 'CLADE', 'REF', 'ALT', 'WT', 'VAR']
    missing_columns = [col for col in required_columns if col not in variants_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{training_params['tokenizer_dir']}/tokenizer.json",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="[SOS]",
        eos_token="[EOS]"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() and dev_params['num_gpus'] > 0 else "cpu")
    logger.info(f"Using device: {device}")
    gpu_settings = detect_gpu_architecture()
    
    logger.info("Initializing model...")
    model = LOLEVE(
        tokenizer=tokenizer,
        num_layers=training_params['num_layers'],
        num_embd=training_params['num_embd'],
        num_heads=training_params['n_head'],
        max_positional_embedding_size=training_params['max_positional_embedding_size'],
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay'],
        embeddings_file=training_params['embeddings_file'],
        model_device=device,
        gpu_settings=gpu_settings,
        use_control_codes=dev_params['use_control_codes']
    )
    
    model.prepare_model(len(tokenizer))
   
    logger.info("Loading model checkpoint...")
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if any(key.startswith("model._orig_mod.") for key in checkpoint['state_dict'].keys()):
        # Handle compiled model state dict
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith("model._orig_mod."):
                new_key = key.replace("model._orig_mod.", "model.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    else:
        # Regular state dict
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    
    experiments = ['all']
    batch_size = min(training_params['batch_size'], len(variants_df))
    
    for exp_type in experiments:
        logger.info(f"\nRunning {exp_type} experiment...")
        variants_data = prep_variants(variants_df, exp_type)
        
        data_module = InferenceSequenceDataModuleLLR(
            batch_size=batch_size,
            sequence_file=None,
            tokenizer=tokenizer,
            val_split=0,
            max_positional_embedding_size=training_params['max_positional_embedding_size'],
            cache=None,
            use_weighted_sampler=False,
            num_embd=training_params['num_embd'],
            model_device=device,
            validation_chromosome=None,
            num_cpus=2
        )
        
        wt_sequences = [v['wt_seq'] for v in variants_data]
        variant_sequences = [v['variant_seq'] for v in variants_data]
        gene_tokens = [v['gene'] for v in variants_data]
        species_tokens = [v['species'] for v in variants_data]
        clade_tokens = [v['clade'] for v in variants_data]
        
        data_module.set_sequences(
            wt_sequences, variant_sequences, gene_tokens, 
            species_tokens, clade_tokens
        )
        
        scores = calculate_llr(model, data_module.val_dataloader(), normalize=normalize)
        # Unzip the tuples into separate lists
        avg_scores, forward_scores = zip(*scores)
        variants_df[f'score_{exp_type}'] = avg_scores
        variants_df[f'score_{exp_type}_forward'] = forward_scores
    
    output_file = f"{output_prefix}_ablation_results.csv"
    logger.info(f"\nSaving results to {output_file}")
    variants_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control code ablation experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--variants', type=str, required=True, help='Path to variants CSV')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to ESM checkpoint')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for output file')
    parser.add_argument('--normalize', action='store_true', help='Normalize scores by length', default=False)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    score_variants_ablation(config, args.variants, args.model_checkpoint, args.output_prefix, args.normalize)
