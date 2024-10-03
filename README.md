# LOL-EVE: Predicting Promoter Variant Effects from Evolutionary Sequences

This repository contains the code and data for the paper "LOL-EVE: Predicting Promoter Variant Effects from Evolutionary Sequences" submitted to ICLR 2024.

## Abstract

Genetic studies reveal extensive disease-associated variation across the human
genome, predominantly in noncoding regions, such as promoters. Quantifying
the impact of these variants on disease risk is crucial to our understanding of
the underlying disease mechanisms and advancing personalized medicine. However, current computational methods struggle to capture variant effects, particularly those of insertions and deletions (indels), which can significantly disrupt
gene expression. To address this challenge, we present LOL-EVE (Language Of
Life across EVolutionary Effects), a conditional autoregressive transformer model
trained on 14.6 million diverse mammalian promoter sequences. Leveraging evolutionary information and proximal genetic context, LOL-EVE predicts indel variant effects in human promoter regions. We introduce three new benchmarks for
indel variant effect prediction in promoter regions, comprising the identification of
causal eQTLs, prioritization of rare variants in the human population, and understanding disruptions of transcription factor binding sites. We find that LOL-EVE
achieves state-of-the-art performance on these tasks, demonstrating the potential
of region-specific large genomic language models and offering a powerful tool for
prioritizing potentially causal non-coding variants in disease studies.

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
    │   ├── gnomad_indel_freq
    │   │   └── plot.ipynb
    │   └── tfbs_disruption
    │       ├── get_consistent_variable_expression.py
    │       ├── get_mutations.py
    │       └── plot.ipynb
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