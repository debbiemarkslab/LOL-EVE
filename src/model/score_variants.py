import argparse
from argparse import ArgumentParser
import logging
import torch
from pytorch_lightning import Trainer
from models import PromEVEModel, InferenceSequenceDataModuleLLR
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import pandas as pd
import itertools as it
from Bio import SeqIO
from Bio.Seq import PromEVEModel
import glob
import os
from collections import Counter
import numpy as np


def prep_variants(variants):

    def create_variant_seq(row):
        
        max_length = 1000
        pos = row['POS']
        ref = row['REF']
        alt = row['ALT'] if row['ALT'] != '.' else ''
        wt_sequence = row['WT_SEQUENCE']
        ref_len = len(ref)
        alt_len = len(alt)
        variant_pos = pos - row['WT_SEQUENCE_START']

        # Skip if the variant sequence would exceed the max_length
        if variant_pos + len(alt) > max_length:
            print(f"Skipping variant at position {pos} for {row['PROMOTER_GENE']} as it exceeds the max length of {max_length} bp.")
            ref_seq = ''
            variant_seq = ''
        else:
            # Adjust the window size based on the variant's position
            left_window = min(variant_pos, (max_length - max(ref_len, alt_len)) // 2)
            right_window = min(len(wt_sequence) - variant_pos - ref_len, max_length - left_window - max(ref_len, alt_len))

            start = max(0, variant_pos - left_window)
            end_ref = min(len(wt_sequence), variant_pos + ref_len + right_window)
            
            ref_seq = wt_sequence[start:end_ref]
            variant_seq = wt_sequence[start:variant_pos] + alt + wt_sequence[variant_pos + ref_len:end_ref]

            # Ensure the sequences are of equal length without adding blanks
            if len(variant_seq) < len(ref_seq):
                variant_seq += ref_seq[len(variant_seq):]
            elif len(variant_seq) > len(ref_seq):
                variant_seq = variant_seq[:len(ref_seq)]

            try:
                assert ' ' not in ref_seq, ref_seq
                assert ' ' not in variant_seq, variant_seq
                assert ref_seq != variant_seq, f'Sequences are the same. Variant pos: {variant_pos}, Ref: {ref}, Alt: {alt}'
                assert len(ref_seq) == len(variant_seq) > 0, f'Sequence empty or unequal length: WT_S:{len(ref_seq)}, ALT_S:{len(variant_seq)}, REF:{ref}, ALT:{alt}'
                assert len(ref_seq) == len(variant_seq) <= 1000, f'Sequence too long: WT_S:{len(ref_seq)}, ALT_S:{len(variant_seq)}, REF:{ref}, ALT:{alt}'
            except AssertionError as e:
                print(f"Assertion failed: {str(e)}")
                print(f"Processing variant: {row['PROMOTER_GENE']} at position {row['POS']}")
                print(f"Variant position in WT_SEQUENCE: {variant_pos}")
                print(f"WT_SEQUENCE length: {len(wt_sequence)}")
                print(f"ref_seq length: {len(ref_seq)}")
                print(f"variant_seq length: {len(variant_seq)}")
                print(f"ref_seq: {ref_seq}")
                print(f"variant_seq: {variant_seq}")
                ref_seq = ''
                variant_seq = ''

        return {
            'wt_seq': ref_seq,
            'variant_seq': variant_seq,
            'gene': row['PROMOTER_GENE'],
            'species': row['SPECIES'],
            'clade': row['CLADE'],
            'REF': ref,
            'ALT': alt,
            'CHROM': row['CHROM'],
            'POS': row['POS']
        }
    
    variants =  variants.apply(create_variant_seq, axis=1).tolist()
    return variants



def calculate_llr(model, tokenizer, dataloader):
    model.eval()

    # Define the complement mapping
    complement_map = {
        tokenizer.vocab['a']: tokenizer.vocab['t'],  # A -> T
        tokenizer.vocab['t']: tokenizer.vocab['a'],  # T -> A
        tokenizer.vocab['c']: tokenizer.vocab['g'],  # C -> G
        tokenizer.vocab['g']: tokenizer.vocab['c']   # G -> C
    }
    start_token = tokenizer.convert_tokens_to_ids('start')
    end_token = tokenizer.convert_tokens_to_ids('end')

    llr_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating LLR")):
            wt_input_ids = batch['wt_input_ids'].to(model.device)
            variant_input_ids = batch['variant_input_ids'].to(model.device)

            # Function to compute log-likelihood for a batch of input sequences
            def compute_log_likelihood(input_ids):
                batch_size, seq_len = input_ids.size()
                
                # Find start and end positions for each sequence in the batch
                start_positions = (input_ids == start_token).nonzero(as_tuple=True)[1]
                end_positions = (input_ids == end_token).nonzero(as_tuple=True)[1]
                

                
                if len(start_positions) != batch_size or len(end_positions) != batch_size:
                    print("  WARNING: Number of start/end tokens doesn't match batch size!")
                    for i, seq in enumerate(input_ids):
                        seq_start = (seq == start_token).nonzero(as_tuple=True)[0]
                        seq_end = (seq == end_token).nonzero(as_tuple=True)[0]

                # Create masks for valid sequences and compute sequence lengths
                valid_mask = (start_positions < end_positions) & (start_positions < seq_len - 1) & (end_positions > 0)
                if not valid_mask.any():
                    print("  ERROR: No valid sequences in this batch!")
                    return torch.full((batch_size,), float('nan'), device=model.device)

                seq_lengths = end_positions[valid_mask] - start_positions[valid_mask] - 1
                max_seq_length = seq_lengths.max().item()

                # Prepare forward sequences
                forward = input_ids[valid_mask]
                forward_start = start_positions[valid_mask]
                forward_end = end_positions[valid_mask]

                # Prepare reverse-complement sequences
                reverse = forward.clone()
                for i, (start, end) in enumerate(zip(forward_start, forward_end)):
                    sequence_part = reverse[i, start+1:end]
                    complemented = torch.tensor([complement_map[n.item()] for n in sequence_part], 
                                                device=reverse.device)
                    reverse[i, start+1:end] = torch.flip(complemented, dims=[0])

                # Pad sequences to max_seq_length
                forward_padded = torch.full((valid_mask.sum(), max_seq_length + 2), 3, 
                                            dtype=forward.dtype, device=forward.device)
                reverse_padded = torch.full((valid_mask.sum(), max_seq_length + 2), 3, 
                                            dtype=reverse.dtype, device=reverse.device)

                for i, (f_start, f_end, r_start, r_end) in enumerate(zip(forward_start, forward_end, forward_start, forward_end)):
                    forward_padded[i, :f_end-f_start+1] = forward[i, f_start:f_end+1]
                    reverse_padded[i, :r_end-r_start+1] = reverse[i, r_start:r_end+1]

                # Calculate logits and labels for forward and reverse sequences
                outputs_forward = model(forward_padded)
                outputs_reverse = model(reverse_padded)
                
                logits_forward = outputs_forward.logits[:, :-1].reshape(-1, outputs_forward.logits.size(-1))
                logits_reverse = outputs_reverse.logits[:, :-1].reshape(-1, outputs_reverse.logits.size(-1))
                
                labels_forward = forward_padded[:, 1:].reshape(-1)
                labels_reverse = reverse_padded[:, 1:].reshape(-1)

                # Compute log-likelihood (negative cross-entropy loss)
                loss_forward = F.cross_entropy(logits_forward, labels_forward, reduction='none', ignore_index=3)
                loss_reverse = F.cross_entropy(logits_reverse, labels_reverse, reduction='none', ignore_index=3)

                # Reshape losses to batch size and sequence length
                loss_forward = loss_forward.view(valid_mask.sum(), -1)
                loss_reverse = loss_reverse.view(valid_mask.sum(), -1)

                # Compute average loss for each sequence
                avg_loss_forward = loss_forward.sum(dim=1) / seq_lengths
                avg_loss_reverse = loss_reverse.sum(dim=1) / seq_lengths

                # Average over both forward and reverse directions
                avg_loss = (avg_loss_forward + avg_loss_reverse) / 2

                # Prepare final result
                log_likelihoods = torch.full((batch_size,), float('nan'), device=model.device)
                log_likelihoods[valid_mask] = -avg_loss  # Negative to represent log-likelihood

                return log_likelihoods

            # Compute log-likelihoods for WT and variant sequences
            try:
                wt_log_likelihood = compute_log_likelihood(wt_input_ids)
                variant_log_likelihood = compute_log_likelihood(variant_input_ids)

                # Calculate LLR
                llr = wt_log_likelihood - variant_log_likelihood 
                llr_results.extend(llr.tolist())
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                llr_results.extend([float('nan')] * len(wt_input_ids))

    return llr_results

def score_variants(model_checkpoint, tokenizer_path, variants, embedding_file, batch_size, prefix):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json",
                                        unk_token="[UNK]",
                                        sep_token="[SEP]",
                                        pad_token="[PAD]",
                                        cls_token="[CLS]")

    variants_data = prep_variants(variants)

    wt_sequences = [v['wt_seq'].lower() if v['wt_seq'] else '' for v in variants_data]
    variant_sequences = [v['variant_seq'].lower() if v['variant_seq'] else '' for v in variants_data]
    gene_tokens = [v['gene'].lower() if v['gene'] else '' for v in variants_data]
    species_tokens = [v['species'].lower() if v['species'] else '' for v in variants_data]
    clade_tokens = [v['clade'].lower() if v['clade'] else '' for v in variants_data]
    ref = [v['REF'] if v['REF'] else '' for v in variants_data]
    alt = [v['ALT'] if v['ALT'] else '' for v in variants_data]
    pos = [v['POS'] if v['POS'] else '' for v in variants_data]
    chrom = [v['CHROM'] if v['CHROM'] else '' for v in variants_data]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_module = InferenceSequenceDataModuleLLR(batch_size, None, tokenizer, .1, 1010, None, None, 768, device, None)
    data_module.set_sequences(wt_sequences, variant_sequences, gene_tokens, species_tokens, clade_tokens, True)

    model = PromEVEModel(tokenizer, num_layers=12, num_embd=768, num_heads=12, max_positional_embedding_size=1010, lr=1e-5, embeddings_file=embedding_file, model_device=device)
    model.prepare_model(data_module.vocab_size)
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model = torch.compile(model)
    val_loader = data_module.val_dataloader()
    llr_per_variant = calculate_llr(model, tokenizer, val_loader)

    # Replace None with blank strings
    llr_per_variant = [str(llr) if llr is not None else '' for llr in llr_per_variant]

    return llr_per_variant


def check_variants(genome_path, variants):

    genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

    for _, row in variants.iterrows():
        ref = str(row['REF']).upper()
        chrom = row['CHROM']

        sequence_pos = abs(row['POS'] - row['WT_SEQUENCE_START'])
        pos = row['POS'] - 1
        
        sequence = row['WT_SEQUENCE']
        print(sequence)
        genome_value = str(genome[chrom][pos:len(ref)+pos].seq).upper()

        assert  genome_value == ref, f'Variant {chrom}:{pos}-{pos+len(ref)} not found in genome, REF: {ref}, Genome: {genome_value}'
        assert sequence[sequence_pos:len(ref)+sequence_pos] == ref, f'Variant not found in sequence, REF:{ref}, sequence:{sequence[sequence_pos:len(ref)+sequence_pos]}, {row}'
    print('Finished validating variants!')

    return variants

def main(variants, checkpoint, tokenizer_path, genome_path, output_file, embedding_file, prefix):
    batch_size = 16
    variants = pd.read_csv(variants)
    #variants = check_variants(genome_path, variants)
    llr_per_variant = score_variants(checkpoint, tokenizer_path, variants, embedding_file, batch_size, prefix)
    
    if not llr_per_variant:
        variants['score'] = ''
    else:
        variants['score'] = llr_per_variant
    
    variants.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score variants")
    parser.add_argument('--variants', type=str, required=True, help='Path to the variants CSV file')
    parser.add_argument('--checkpoint_esm', type=str, required=True, help='Path to the ESM checkpoint')
    parser.add_argument('--tokenizer_path_esm', type=str, required=True, help='Path to the ESM tokenizer')
    parser.add_argument('--genome_path', type=str, required=True, help='Path to the genome FASTA file')
    parser.add_argument('--output_file', type=str, required=True, help='Path for the output CSV file')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to the embedding file')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files')

    args = parser.parse_args()

    main(args.variants, args.checkpoint_esm, args.tokenizer_path_esm, args.genome_path, args.output_file, args.embedding_file, args.prefix)

