# LOL-EVE: A Genomic Language Model for Zero-Shot Prediction of Promoter Variant Effects

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/Marks-lab/LOL-EVE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for the paper "A Genomic Language Model for Zero-Shot Prediction of Promoter Variant Effects" accepted at MLCB 2025.

## Abstract

Disease-associated genetic variants occur extensively across the human genome, predominantly in noncoding regions like promoters. While crucial for understanding disease mechanisms, current methods struggle to predict effects of insertions and deletions (indels) that can disrupt gene expression. We present LOL-EVE (Language Of Life for Evolutionary Variant Effects), a conditional autoregressive transformer trained on 13.6 million mammalian promoter sequences. By leveraging evolutionary patterns and genetic context, LOL-EVE enables zero-shot prediction of indel effects in human promoters. We introduce three new benchmarks for promoter indel prediction: ultra rare variant prioritization, causal eQTL identification, and transcription factor binding site disruption analysis. LOL-EVE's state of the art performance across these tasks suggests the potential of region-specific genomic language models for identifying causal non-coding variants in disease studies.

## 🤗 Hugging Face Integration

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Marks-lab/LOL-EVE')
model = AutoModelForCausalLM.from_pretrained('Marks-lab/LOL-EVE', trust_remote_code=True)

# Basic DNA sequence
sequence = "[MASK] [MASK] [MASK] [SOS]ATGCTAGCTAGCTAGCTAGCTA[EOS]"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

# With control codes (recommended)
control_sequence = "brca1 human primate [SOS] ATGCTAGCTAGCTAGCTAGCTA [EOS]"
inputs = tokenizer(control_sequence, return_tensors="pt")
outputs = model(**inputs)
```

### Input Format

The model expects sequences in the format:
```
gene species clade [SOS] sequence [EOS]
```

Where:
- `gene`: Gene name (e.g., "brca1", "tp53")
- `species`: Species name (e.g., "human", "mouse") 
- `clade`: Clade information (e.g., "primate", "mammal")
- `[SOS]`: Start of sequence token
- `sequence`: DNA sequence (A, T, G, C)
- `[EOS]`: End of sequence token

### Model Specifications

- **Architecture**: Causal Language Model (CTRL-based)
- **Layers**: 12 transformer layers
- **Hidden size**: 768 dimensions
- **Attention heads**: 12
- **Vocabulary size**: 39,378 tokens
- **Max sequence length**: 1,007 tokens
- **Position embeddings**: Adaptive local position embeddings

### Key Features

- **Large vocabulary**: 39,378 tokens including DNA bases, control codes, and special tokens
- **Control code integration**: Incorporates gene, species, and clade information
- **Protein context**: Uses pre-trained ESM embeddings for gene-specific understanding
- **Flexible input format**: Supports both basic DNA sequences and control code sequences

## Repository Structure

```bash
├── LICENSE
├── README.md
├── requirement.txt
├── DEPLOYMENT_GUIDE.md
├── src/
│   ├── benchmarks/
│   │   ├── causal_eqtls/
│   │   │   └── plot.ipynb
│   │   ├── gnomad_ultra_rare/
│   │   │   └── plot.ipynb
│   │   └── tfbs_disruption/
    |           COMING SOON
│   ├── configs/
│   │   └── example_config.json
│   ├── core/
│   │   ├── benchmark_analyzer.py
│   │   ├── benchmark_callback.py
│   │   ├── data.py
│   │   ├── embeddings.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── scoring.py
│   │   └── utils.py
│   └── scripts/
│       ├── score_sequences.py
│       └── train.py
└── huggingface/
    ├── README.md
    ├── example_usage.py
    ├── modeling_loleve.py
    ├── requirements.txt
    └── setup_huggingface.py
```

## Installation

### Prerequisites

This project requires CUDA. If CUDA is not already set up on your system, you may need to load it as a module or install it separately:

```bash
# On some systems, you might use:
module load cuda

# On others, you might need to specify a version:
module load cuda/11.8
module load shared cuda11.8/toolkit/11.8.0

# Check with your system administrator for the correct command if unsure.
```

### Setup

```bash
git clone https://github.com/Marks-lab/LOL-EVE
cd LOL-EVE
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python src/scripts/train.py --config src/configs/example_config.json
```

### Generate Tokenizer

```bash
python src/scripts/generate_vocab.py --input sequences.parquet --output path/to/tokenizer
```

### Score Variants

#### Using Local Checkpoint
```bash
python src/scripts/score_sequences.py \
    --variants $INPUT_FILE \
    --checkpoint $CHECKPOINT \
    --tokenizer_path $TOKENIZER_PATH \
    --genome_path $GENOME_PATH \
    --output_file $OUTPUT_FILE \
    --embedding_file $EMBEDDING_FILE
```

#### Using Hugging Face Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('Marks-lab/LOL-EVE')
model = AutoModelForCausalLM.from_pretrained('Marks-lab/LOL-EVE', trust_remote_code=True)

def score_variants_hf(variants_df, gene, species, clade):
    """
    Score variants using the Hugging Face model.
    
    Args:
        variants_df: DataFrame with columns ['sequence', 'variant_sequence']
        gene: Gene name (e.g., 'brca1')
        species: Species name (e.g., 'human')
        clade: Clade information (e.g., 'primate')
    
    Returns:
        DataFrame with added 'score' column
    """
    scores = []
    
    for _, row in variants_df.iterrows():
        # Create control code sequences
        ref_seq = f"{gene} {species} {clade} [SOS] {row['sequence']} [EOS]"
        var_seq = f"{gene} {species} {clade} [SOS] {row['variant_sequence']} [EOS]"
        
        # Tokenize sequences
        ref_inputs = tokenizer(ref_seq, return_tensors="pt")
        var_inputs = tokenizer(var_seq, return_tensors="pt")
        
        # Get model outputs
        with torch.no_grad():
            ref_outputs = model(**ref_inputs)
            var_outputs = model(**var_inputs)
            
            # Calculate log-likelihood scores
            ref_logits = ref_outputs.logits[0, :-1]  # Exclude last token
            var_logits = var_outputs.logits[0, :-1]
            
            ref_tokens = ref_inputs['input_ids'][0, 1:]  # Exclude first token
            var_tokens = var_inputs['input_ids'][0, 1:]
            
            # Calculate sequence likelihood
            ref_score = torch.nn.functional.cross_entropy(ref_logits, ref_tokens, reduction='sum')
            var_score = torch.nn.functional.cross_entropy(var_logits, var_tokens, reduction='sum')
            
            # Score is the difference (higher = more deleterious)
            score = (var_score - ref_score).item()
            scores.append(score)
    
    variants_df['score'] = scores
    return variants_df

# Example usage
variants = pd.DataFrame({
    'sequence': ['ATGCTAGCTAGCTAGCTAGCTA', 'ATGCTAGCTAGCTAGCTAGCTA'],
    'variant_sequence': ['ATGCTAGCTAGCTAGCTAGCTA', 'ATGCTAGCTAGCTAGCTAGCTA']  # Example variants
})

scored_variants = score_variants_hf(variants, gene='brca1', species='human', clade='primate')
print(scored_variants)
```

### Benchmarks

All statistical analysis and plotting code is found for each respective benchmark in `/src/benchmarks`:
- **Ultra-rare variant prioritization**: `gnomad_ultra_rare/`
- **Causal eQTL identification**: `causal_eqtls/`
- **Transcription factor binding site disruption**: `tfbs_disruption/`

## Datasets

- **[LOL-EVE-UltraRare](https://huggingface.co/datasets/Marks-lab/LOL-EVE-UltraRare)** - Ultra-rare variant benchmark dataset
- **[LOL-EVE-eQTL_benchmark](https://huggingface.co/datasets/Marks-lab/LOL-EVE-eQTL_benchmark)** - eQTL benchmark dataset
- **Training data**: Available on Zenodo (link TBD)


## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{loleve2025,
  title={A Genomic Language Model for Zero-Shot Prediction of Promoter Variant Effects},
  author={[Authors]},
  journal={MLCB 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns, please open an issue in this repository or contact [courtney.a.shearer@gmail.com].
