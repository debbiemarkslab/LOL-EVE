import json
import os
import torch
from argparse import ArgumentParser
from transformers import PreTrainedTokenizerFast
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import wandb
from models import SequenceDataModule, LOLEVE

# Set environment variable to disable tokenizers' parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages


def detect_gpu_architecture():
    """Detect the GPU architecture and return appropriate training settings."""
    if not torch.cuda.is_available():
        return {
            'precision': '32-true',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    
    gpu_name = torch.cuda.get_device_name().lower()
    
    if 'h100' in gpu_name:
        print("H100 GPU detected - using 32-bit precision for stability")
        return {
            'precision': 'bf16-mixed',  # More stable for H100
            'compile_model': True,
            'initial_lr': 1e-5,  # Lower learning rate for stability
            'gradient_clip_val': 1.0
        }
    elif 'a100' in gpu_name:
        print("A100 GPU detected - using mixed precision")
        return {
            'precision': 'bf16-mixed',
            'compile_model': True,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    else:  # Default for other GPUs (including L40, V100, etc)
        print(f"GPU detected: {gpu_name} - using default settings")
        return {
            'precision': 'bf16-mixed',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }

def main(config_file, checkpoint=None):
    """Main training function with GPU train variations and CPU test mode support."""
    with open(config_file, 'r') as fp:
        config = json.load(fp)
    print(f'Checkpoint: {checkpoint}')

    # Unpack configuration
    training_parameters = config['training_parameters']
    logging_parameters = config['logging_parameters']
    development_parameters = config['development_parameters']
    
    # Override GPU settings if in test mode
    if development_parameters['test_on']:
        print("Running in CPU test mode")
        development_parameters['num_gpus'] = 0
        training_parameters['batch_size'] = 1
        gpu_settings = {
            'precision': '32-true',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    else:
        gpu_settings = detect_gpu_architecture()
    
    batch_size = training_parameters['batch_size']
    sequence_file = training_parameters['sequences']
    tokenizer_dir = training_parameters['tokenizer_dir']
    val_split = training_parameters['val_split']
    max_positional_embedding_size = training_parameters['max_positional_embedding_size']
    validation_chromosome = training_parameters['validation_chromosome']
    num_embd = training_parameters['num_embd']
    num_cpus = training_parameters['num_cpus']
    use_control_codes = development_parameters['use_control_codes']
    # Initialize tokenizer with special tokens
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="[SOS]",  # Add Beginning of Sequence token
        eos_token="[EOS]"   # Add End of Sequence token

    )
    
    # Add additional special tokens
    special_tokens = ['[MASK]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})


    # Determine device configuration
    gpus = development_parameters['num_gpus'] if torch.cuda.is_available() and not development_parameters['test_on'] else 0
    device = torch.device("cpu") if development_parameters['test_on'] else torch.device("cuda" if torch.cuda.is_available() and gpus > 0 else "cpu")
    print(f"Using device: {device}")

    # Set float32 precision only if using GPU
    if not development_parameters['test_on'] and torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Initialize data module
    data_module = SequenceDataModule(
        batch_size=batch_size,
        sequence_file=sequence_file,
        tokenizer=tokenizer,
        val_split=val_split,
        max_positional_embedding_size=max_positional_embedding_size,
        cache=logging_parameters['scratch'],
        use_weighted_sampler=training_parameters['use_weighted_sampler'],
        num_embd=num_embd,
        model_device=device,
        validation_chromosome=validation_chromosome,
        num_cpus=num_cpus
        )
    
    data_module.prepare_data()
    data_module.setup(stage='fit')

    
    # Initialize model
    model = LOLEVE(
        tokenizer=tokenizer,
        num_layers=training_parameters['num_layers'],
        num_embd=num_embd,
        num_heads=training_parameters['n_head'],
        max_positional_embedding_size=max_positional_embedding_size,
        lr=training_parameters['lr'],
        weight_decay=training_parameters['weight_decay'],
        embeddings_file=training_parameters['embeddings_file'],
        model_device=device,
        gpu_settings=gpu_settings,
        use_control_codes=use_control_codes
    )
    model.prepare_model(data_module.vocab_size)

    using_validation = data_module.val_dataset is not None

    # Configure WandB logger if enabled
   
    run_id = logging_parameters['run_id']

    if run_id:
        wandb_logger = WandbLogger(project='promEVE', id=run_id,resume='must') if development_parameters['wandb_on'] else None
    else:
        wandb_logger = WandbLogger(project='promEVE') if development_parameters['wandb_on'] else None
   
    checkpoint_save_validation_run_step = 10000 

    if using_validation:
        checkpoint_callback = ModelCheckpoint(
            dirpath=logging_parameters['model_output'],
            filename='model_epoch_{epoch:02d}-{val_all_control_perplexity_epoch:.4f}',
            save_top_k=3,
            every_n_train_steps=checkpoint_save_validation_run_step,
            monitor='val_all_control_perplexity_epoch',
            mode='min',
            save_weights_only=False
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=logging_parameters['model_output'],
            filename='model_step_{step}-{train_perplexity:.4f}',  # Added perplexity to filename
            save_top_k=3,
            every_n_train_steps=checkpoint_save_validation_run_step,
            monitor='train_perplexity',  # Monitor training perplexity
            mode='min',  # Lower perplexity is better
            save_weights_only=False
        )
    
    # Set up training devices
    devices = 1 if gpus == 0 else gpus
    accelerator = 'cpu' if development_parameters['test_on'] else ('gpu' if gpus > 0 else 'cpu')
    print(f'Using {gpus} GPUs!' if gpus > 0 else 'Using CPU mode')

    # Base trainer arguments
    trainer_kwargs = {
        'max_steps': training_parameters['total_steps'],
        'log_every_n_steps': 100,
        'logger': wandb_logger,
        'devices': devices,
        'accelerator': accelerator,
        'callbacks': [checkpoint_callback],
    }

    # Add validation settings if using validation
    if using_validation:
        trainer_kwargs.update({
            'check_val_every_n_epoch': None,
            'val_check_interval': checkpoint_save_validation_run_step,
            'num_sanity_val_steps': 2
        })
    else:
        trainer_kwargs.update({
            'limit_val_batches': 0,  # Disable validation,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': None,
            'val_check_interval': None
        })
    
    # Add GPU-specific settings only if not in test mode
    if not development_parameters['test_on']:
        trainer_kwargs.update({
            'strategy': DDPStrategy(find_unused_parameters=False),
            'precision': gpu_settings['precision'],
            'gradient_clip_val': gpu_settings['gradient_clip_val'],
            'gradient_clip_algorithm': 'norm',
            'accumulate_grad_batches': 4 if gpu_settings['precision'] == '32-true' else 1
        })
    
    trainer = Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint)
    
    if development_parameters['wandb_on']:
        wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser(description="Train a CTRL model with PyTorch Lightning")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training')
    args = parser.parse_args()
    main(args.config, args.checkpoint)
