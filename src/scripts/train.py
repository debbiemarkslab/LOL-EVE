#!/usr/bin/env python3
import json
import os
import logging
import torch
from argparse import ArgumentParser
from transformers import PreTrainedTokenizerFast
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import wandb

# Import local modules
from core.models import LOLEVE
from core.data import SequenceDataModule
from core.utils import detect_gpu_architecture
from core.benchmark_utils import BenchmarkLogger  # Import the benchmark logger
from core.benchmark_callback import BenchmarkCallback

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages

# Configure logging
logger = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main(config_file, checkpoint=None, benchmarks='None'):
    """
    Main training function with benchmark monitoring
    
    Args:
        config_file: Path to configuration JSON
        checkpoint: Optional path to checkpoint for resuming training
        benchmark_data_dir: Directory containing benchmark data
        benchmark_log_dir: Directory to save benchmark results
        benchmark_freq: Frequency of benchmark evaluation in steps
    """
    seed_everything(42)  # Set seed for reproducibility
    
    # Load configuration
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as fp:
        config = json.load(fp)

    # Use checkpoint from command line if provided, otherwise check config
    if checkpoint is None and 'checkpoint_path' in config['training_parameters']:
        checkpoint = config['training_parameters']['checkpoint_path']
        logger.info(f"Using checkpoint from config: {checkpoint}")
    
    # Unpack configuration
    training_params = config['training_parameters']
    logging_params = config['logging_parameters']
    dev_params = config['development_parameters']

    # Override GPU settings if in test mode
    if dev_params.get('test_on', False):
        logger.info("Running in CPU test mode")
        dev_params['num_gpus'] = 0
        training_params['batch_size'] = 1
        gpu_settings = {
            'precision': '32-true',
            'compile_model': False,
            'initial_lr': 1e-4,
            'gradient_clip_val': 1.0
        }
    else:
        gpu_settings = detect_gpu_architecture()
    
    # Initialize tokenizer with special tokens
    logger.info(f"Initializing tokenizer from {training_params['tokenizer_dir']}")
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
    
    # Add additional special tokens if needed
    special_tokens = ['[MASK]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # Determine device configuration
    gpus = dev_params['num_gpus'] if torch.cuda.is_available() and not dev_params['test_on'] else 0
    device = torch.device("cpu") if dev_params['test_on'] else torch.device("cuda" if torch.cuda.is_available() and gpus > 0 else "cpu")
    logger.info(f"Using device: {device}, GPUs: {gpus}")
    
    # Set float32 precision only if using GPU
    if not dev_params['test_on'] and torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Initialize data module
    logger.info("Initializing data module")
    data_module = SequenceDataModule(
        batch_size=training_params['batch_size'],
        sequence_file=training_params['sequences'],
        tokenizer=tokenizer,
        val_split=training_params['val_split'],
        max_positional_embedding_size=training_params['max_positional_embedding_size'],
        cache=logging_params.get('scratch', None),
        use_weighted_sampler=training_params.get('use_weighted_sampler', False),
        validation_chromosome=training_params.get('validation_chromosome', None),
        num_cpus=training_params.get('num_cpus', 4)
    )
    
    # Prepare data
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    # Initialize model
    logger.info("Initializing LOLEVE model")
    model = LOLEVE(
        tokenizer=tokenizer,
        num_layers=training_params['num_layers'],
        num_embd=training_params['num_embd'],
        num_heads=training_params['n_head'],
        max_positional_embedding_size=training_params['max_positional_embedding_size'],
        lr=training_params['lr'],
        weight_decay=training_params.get('weight_decay', 0.0),
        embeddings_file=training_params['embeddings_file'],
        use_control_codes=dev_params['use_control_codes'],
        model_device=device,
        gpu_settings=gpu_settings,
        position_embedding_type=training_params['position_embedding_type']
    )
    
    # Initialize model parameters
    model.prepare_model(len(tokenizer))
    
    # Configure WandB logger if enabled
    if dev_params.get('wandb_on', False):
        run_id = dev_params.get('run_id')
        if run_id:
            wandb_logger = WandbLogger(
                project=logging_params.get('project', 'LOL-EVE'),
                id=run_id,
                resume='must'
            )
        else:
            wandb_logger = WandbLogger(
                project=logging_params.get('project', 'LOL-EVE')
            )
    else:
        wandb_logger = None
    
    # Configure model checkpointing
    using_validation = data_module.val_dataset is not None
    checkpoint_dir = logging_params.get('model_output', './models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Add checkpoint callback
    if using_validation:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model_epoch_{epoch:02d}-{val_all_control_perplexity_epoch:.4f}',
            save_top_k=3,
            every_n_train_steps=training_params.get('checkpoint_steps', 10000),
            monitor='val_no_ablation_perplexity_epoch',
            mode='min',
            save_weights_only=False
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model_step_{step}-{train_perplexity:.4f}',
            save_top_k=3,
            every_n_train_steps=training_params.get('checkpoint_steps', 10000),
            monitor='train_perplexity',
            mode='min',
            save_weights_only=False
        )
    
    callbacks.append(checkpoint_callback)
    
    # Add learning rate monitor
    if wandb_logger:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    benchmark_log_dir = logging_params["benchmark_log_dir"]
    benchmark_freq = logging_params["benchmark_freq"]

    if 'none' not in benchmarks:
        benchmark_log_dir = logging_params["benchmark_log_dir"]
        
        # Use a different approach to check if this is the main process
        import torch.distributed as dist
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        # Create directory from the main process only
        if is_main_process:
            os.makedirs(benchmark_log_dir, exist_ok=True)
            logger.info(f"Created benchmark log directory: {benchmark_log_dir}")
        
        # If distributed, make sure all processes see the created directory
        if gpus > 1 and dist.is_initialized():
            dist.barrier()  # Wait for rank 0 to create directory
        
        # Then initialize benchmark logger and callback
        logger.info(f"Setting up benchmark monitoring with frequency {benchmark_freq} steps")
        
        # Create benchmark logger
        benchmark_logger = BenchmarkLogger(
            model=model,
            tokenizer=tokenizer,
            config=config,  # Pass the full config
            log_dir=benchmark_log_dir,
            device=device,
            log_freq=benchmark_freq,
            benchmarks_to_run=benchmarks
        )
        
        # Load tail variant subset for training benchmarks
        rare_dataset_path = config.get("logging_parameters", {}).get("benchmark_datasets", {}).get("rare", {}).get("path")
        if rare_dataset_path and 'rare' in benchmarks:
            print('Loading Tail variants')
            benchmark_logger.load_tail_variant_subset(
                full_dataset_path=rare_dataset_path,
                loleve_column='LOL-EVE_AF',  # Or whatever your score column is named
                percentile=1.0,
                save_path=os.path.join(benchmark_log_dir, "rare_tail_subset.csv")
            )
            
        # Create benchmark callback
        benchmark_callback = BenchmarkCallback(
            benchmark_logger=benchmark_logger,
            eval_every_n_steps=benchmark_freq,
            save_dir=benchmark_log_dir
        )
        
        # Add to callbacks list
        callbacks.append(benchmark_callback)
    else:
        logger.info("Benchmark evaluation disabled")
        
        # No callback needed!
    # Set up training devices
    devices = 1 if gpus == 0 else gpus
    accelerator = 'cpu' if dev_params['test_on'] else ('gpu' if gpus > 0 else 'cpu')
    logger.info(f"Using {devices} {'GPU' if accelerator == 'gpu' else 'CPU'}{'' if devices <= 1 else 's'}")
    
    # Configure trainer
    trainer_kwargs = {
        'max_steps': training_params.get('total_steps', None),
        'max_epochs': training_params.get('epochs', None) if not training_params.get('total_steps') else None,
        'log_every_n_steps': 100,
        'logger': wandb_logger,
        'devices': devices,
        'accelerator': accelerator,
        'callbacks': callbacks,
    }
    
    # Add validation settings if using validation
    if using_validation:
        if training_params.get('total_steps'):
            val_check_interval = min(
                training_params.get('checkpoint_steps', 10000),
                training_params.get('total_steps') // 10
            )
            trainer_kwargs.update({
                'check_val_every_n_epoch': None,
                'val_check_interval': val_check_interval,
                'num_sanity_val_steps': 2
            })
        else:
            trainer_kwargs.update({
                'check_val_every_n_epoch': 1,
                'val_check_interval': None,
                'num_sanity_val_steps': 2
            })
    else:
        trainer_kwargs.update({
            'limit_val_batches': 0,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': None,
            'val_check_interval': None
        })
    
    # Add GPU-specific settings only if not in test mode
    if not dev_params['test_on']:
        trainer_kwargs.update({
            'strategy': DDPStrategy(find_unused_parameters=True),
            'precision': gpu_settings['precision'],
            'gradient_clip_val': gpu_settings['gradient_clip_val'],
            'gradient_clip_algorithm': 'norm',
            'accumulate_grad_batches': training_params.get('accumulate_grad_batches', 1)
        })
    
    # Initialize trainer
    trainer = Trainer(**trainer_kwargs)
    
    # Train model
    logger.info("Starting training")

    # print("Checking parameter gradients:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint)
    
    # Finish WandB run
    if dev_params.get('wandb_on', False) and wandb.run:
        wandb.finish()
    
    logger.info("Training complete")
    return trainer


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a LOLEVE model with PyTorch Lightning and benchmark monitoring")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--benchmarks', nargs='+', choices=['all', 'eqtl', 'rare', 'tfbs', 'none'], 
                        default=['none'], help='Which benchmarks to run (default: all)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        main(
            args.config, 
            args.checkpoint,
            args.benchmarks
        )
    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise
