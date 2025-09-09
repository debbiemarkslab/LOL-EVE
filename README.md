# LOL-EVE: A Genomic Language Model for Zero-Shot Prediction of Promoter Variant Effects

This repository contains the code and data for the paper "A Genomic Language Model for Zero-Shot Prediction of Promoter Variant Effects" accepted at MLCB 2025.

## Abstract

Disease-associated genetic variants occur extensively
across the human genome, predominantly
in noncoding regions like promoters. While crucial
for understanding disease mechanisms, current
methods struggle to predict effects of insertions
and deletions (indels) that can disrupt
gene expression. We present LOL-EVE (Language
Of Life for Evolutionary Variant Effects),
a conditional autoregressive transformer trained
on 13.6 million mammalian promoter sequences.
By leveraging evolutionary patterns and genetic
context, LOL-EVE enables zero-shot prediction
of indel effects in human promoters. We introduce
three new benchmarks for promoter indel
prediction: ultra rare variant prioritization, causal
eQTL identification, and transcription factor binding
site disruption analysis. LOL-EVE’s state of
the art performance across these tasks suggests
the potential of region-specific genomic language
models for identifying causal non-coding variants
in disease studies.

## Repository Structure

```bash
.
├── LICENSE
├── README.md
├── requirement.txt
└── src
    ├── benchmarks
    │   ├── causal_eqtls
    │   │   └── plot.ipynb
    │   ├── gnomad_ultra_rare
    │   │   └── plot_icml.ipynb
    │   └── tfbs_disruption
    │       ├── get_mutations.py
    │       └── plot_icml.ipynb
    └── model
        ├── generate_tokenizer.py
        ├── models.py
        ├── sample_config.json
        ├── score_variants.py
        └── train.py
```

## Installation

To set up the project environment:

```bash
git clone [ANONYMOUS_REPO_URL]
cd lol-eve
```

### CUDA Setup
This project requires CUDA. If CUDA is not already set up on your system, you may need to load it as a module or install it separately. The exact command may vary depending on your system:

```bash
# On some systems, you might use:
module load cuda

# On others, you might need to specify a version:
module load cuda/11.8
module load shared cuda11.8/toolkit/11.8.0

# Check with your system administrator for the correct command if unsure.
```

```bash
pip install -r requirements.txt
```

### Usage

Benchmarks:
- all statistical analysis and plotting code is found for each respecive benchmark in /src/benchmarks

Generate Tokenizer:

```bash
python generate_tokenizer.py --input sequences.parquet --output path/to/tokenizer
```

Training a model:

```bash
python model/train.py --config sample_config.json
```

Score a model:

```bash
python score_variants.py \
    --variants $INPUT_FILE \
    --checkpoint $CHECKPOINT \
    --tokenizer_path $TOKENIZER_PATH \
    --genome_path $GENOME_PATH \
    --output_file $OUTPUT_FILE \
    --embedding_file $EMBEDDING_FILE
```

### Datasets
 
Zenodo link: XXXX


### Citation
If you find this work useful for your research, please consider citing our paper: XXXX

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Contact
For any questions or concerns, please open an issue in this repository.
