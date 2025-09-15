#!/usr/bin/env python3
"""
Example usage script for LOL-EVE model.
This script demonstrates how to load and use the LOL-EVE model for genomic sequence analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("🧬 LOL-EVE Example Usage")
    print("=" * 40)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Marks-lab/LOL-EVE')
    model = AutoModelForCausalLM.from_pretrained('Marks-lab/LOL-EVE', trust_remote_code=True)
    print("✅ Model loaded successfully!")
    
    # Example 1: Basic DNA sequence
    print("\n1. Basic DNA Sequence Analysis")
    print("-" * 30)
    basic_sequence = "[MASK] [MASK] [MASK] [SOS]ATGCTAGCTAGCTAGCTAGCTA[EOS]"
    print(f"Input: {basic_sequence}")
    
    inputs = tokenizer(basic_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Sequence length: {outputs.logits.shape[1]} tokens")
    
    # Example 2: Control code sequence (recommended)
    print("\n2. Control Code Sequence Analysis")
    print("-" * 30)
    control_sequence = "brca1 human primate [SOS] ATGCTAGCTAGCTAGCTAGCTA [EOS]"
    print(f"Input: {control_sequence}")
    
    inputs = tokenizer(control_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Sequence length: {outputs.logits.shape[1]} tokens")
    
    # Example 3: Different gene
    print("\n3. Different Gene Analysis")
    print("-" * 30)
    tp53_sequence = "tp53 human primate [SOS] GATCGATCGATCGATCGATCGA [EOS]"
    print(f"Input: {tp53_sequence}")
    
    inputs = tokenizer(tp53_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Sequence length: {outputs.logits.shape[1]} tokens")
    
    print("\n" + "=" * 40)
    print("🎉 All examples completed successfully!")
    print("The model is ready for your genomic analysis tasks.")

if __name__ == "__main__":
    main()
