"""
Common utility functions for LOL-EVE.
"""
import json
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from datasets import load_dataset 
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


def setup_logging(name, level=logging.INFO):
    """Configure logging"""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def detect_gpu_architecture(logger_name="train"):
    """Detect GPU architecture and return appropriate training settings.
    
    Args:
        logger_name: Name of the logger to use (default: "train")
    
    Returns:
        Dict with GPU-specific settings
    """
    # Get the same logger that was used in train.py
    logger = logging.getLogger(logger_name)
    
    if not torch.cuda.is_available():
        return {
            'precision': '32-true',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    
    gpu_name = torch.cuda.get_device_name().lower()
    
    if 'h100' in gpu_name:
        logger.info("H100 GPU detected - using BF16 mixed precision")
        return {
            'precision': 'bf16-mixed',
            'compile_model': True,
            'initial_lr': 5e-5,
            'gradient_clip_val': 0.5
        }
    elif 'a100' in gpu_name:
        logger.info("A100 GPU detected - using BF16 mixed precision")
        return {
            'precision': 'bf16-mixed',
            'compile_model': True,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    else:
        logger.info(f"GPU detected: {gpu_name} - using default settings")
        return {
            'precision': 'bf16-mixed',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }

def load_model_checkpoint(model, checkpoint_path):
    """Load model checkpoint and handle compiled model state
    
    Args:
        model: The model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        The loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    # Handle compiled model state dict if needed
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # For older checkpoints
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # Handle compiled model keys
        if key.startswith("model._orig_mod."):
            new_key = key.replace("model._orig_mod.", "model.")
            new_state_dict[new_key] = value
        # Handle adaptive_pos_embedding vs position_embedding
        elif 'adaptive_pos_embedding' in key:
            new_key = key.replace('adaptive_pos_embedding', 'position_embedding')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Check for missing keys before loading
    missing_keys = set(model.state_dict().keys()) - set(new_state_dict.keys())
    unexpected_keys = set(new_state_dict.keys()) - set(model.state_dict().keys())
    
    if missing_keys or unexpected_keys:
        print(f"Warning: Some keys are still mismatched after remapping:")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def load_model(config_path, checkpoint_path, device):
    """Load model from checkpoint
    
    Args:
        config_path: Path to model config JSON
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer_path = config['training_parameters']['tokenizer_dir']
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
    
    # Set up GPU settings
    gpu_settings = {
        'precision': 'bf16-mixed',
        'compile_model': False,
        'initial_lr': config['training_parameters'].get('lr', 1e-4),
        'gradient_clip_val': 1.0
    }
    
    # Import locally to avoid circular imports
    from .models import LOLEVE
    
    # Initialize model
    model = LOLEVE(
        tokenizer=tokenizer,
        num_layers=config['training_parameters']['num_layers'],
        num_embd=config['training_parameters']['num_embd'],
        num_heads=config['training_parameters']['n_head'],
        max_positional_embedding_size=config['training_parameters']['max_positional_embedding_size'],
        lr=config['training_parameters']['lr'],
        weight_decay=config['training_parameters'].get('weight_decay', 0.0),
        embeddings_file=config['training_parameters']['embeddings_file'],
        model_device=device,
        gpu_settings=gpu_settings
    )
    
    # Initialize model parameters
    model.prepare_model(len(tokenizer))
    
    # Load checkpoint
    load_model_checkpoint(model, checkpoint_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def plot_results(results, output_dir, output_prefix):
    """Create visualization of evaluation results
    
    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save plots
        output_prefix: Prefix for output filenames
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-darkgrid')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot loss
    exp_names = list(results.keys())
    loss_values = [results[exp]['loss'] for exp in exp_names]
    perplexity_values = [results[exp]['perplexity'] for exp in exp_names]
    accuracy_values = [results[exp]['accuracy'] for exp in exp_names]
    
    # Loss plot
    axes[0].bar(exp_names, loss_values, color='skyblue')
    axes[0].set_title('Loss by Experiment')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticklabels(exp_names, rotation=45, ha='right')
    
    # Perplexity plot
    axes[1].bar(exp_names, perplexity_values, color='lightgreen')
    axes[1].set_title('Perplexity by Experiment')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_xticklabels(exp_names, rotation=45, ha='right')
    
    # Accuracy plot
    axes[2].bar(exp_names, accuracy_values, color='salmon')
    axes[2].set_title('Accuracy by Experiment')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xticklabels(exp_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_results.png", dpi=300)
    plt.close()
    
    # Create distribution plots for each metric
    for metric, values in [('loss', 'all_losses'), ('perplexity', 'all_perplexities'), ('accuracy', 'all_accuracies')]:
        plt.figure(figsize=(10, 6))
        
        for exp_name in results:
            sns.kdeplot(results[exp_name][values], label=exp_name)
        
        plt.title(f'Distribution of {metric.capitalize()} Values')
        plt.xlabel(metric.capitalize())
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f"{output_dir}/{output_prefix}_{metric}_distribution.png", dpi=300)
        plt.close()

def generate_vocab(input_path, output_path, tokenizer_template_path=None):
    """Generate vocabulary from sequence data
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save tokenizer
        tokenizer_template_path: Path to tokenizer template (optional)
    """
    from datasets import load_dataset
    import torch
    
    logger = setup_logging("generate_vocab")
    
    # Use default template path if not provided
    if tokenizer_template_path is None:
        dirname = Path(__file__).parent.parent
        tokenizer_template_path = f"{dirname}/resources/tokenizer_template"
    
    logger.info(f"Using tokenizer template: {tokenizer_template_path}")
    
    # Initialize tokenizer with special tokens
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_template_path,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="[SOS]",
        eos_token="[EOS]"
    )
    
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    
    # Collect unique values
    logger.info("Collecting unique tokens")
    new_values = set()
    for row in tqdm(dataset, desc="Processing dataset"):
        new_values.update(set(row['sequence']))
        new_values.add(row['gene'])
        new_values.add(row['species'])
        new_values.add(row['clade'])
    
    # Add tokens to vocabulary
    logger.info(f"Adding {len(new_values)} tokens to vocabulary")
    tokenizer.add_tokens(sorted(list(new_values)))
    
    # Add start/end tokens if not already present
    for token in ['start', 'end']:
        if token not in tokenizer.vocab:
            tokenizer.add_tokens(token)
    
    # Save tokenizer
    logger.info(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)
    
    return tokenizer