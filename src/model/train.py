import json
import os
import torch
from torch import nn
import numpy as np
from argparse import ArgumentParser
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from models import SequenceDataModule, PromEVEModel
import wandb
from pytorch_lightning.strategies import DDPStrategy

# Set environment variable to disable tokenizers' parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_file, checkpoint=None):

    with open(config_file, 'r') as fp:
        config = json.load(fp)
    print(f'Checkpoint: {checkpoint}')

    training_parameters = config['training_parameters']
    logging_parameters = config['logging_parameters']
    development_parameters = config['development_parameters']
    batch_size, epochs, lr, max_positional_embedding_size,val_split, validation_chromosome, num_layers, num_embd, n_head, sequence_file, tokenizer_dir, embeddings_file, use_weighted_sampler = training_parameters.values()
    _, model_output, cache = logging_parameters.values()
    test_on, wandb_on, num_gpus = development_parameters.values()

    tokenizer =  PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
                                                 unk_token="[UNK]",
                                                 sep_token="[SEP]",
                                                 pad_token="[PAD]",
                                                 cls_token="[CLS]")

    # Determine the device to use
    gpus = num_gpus if torch.cuda.is_available() and test_on == 0 else 0
    device = torch.device("cuda" if torch.cuda.is_available() and gpus > 0 else "cpu")
    print(device)


    data_module = SequenceDataModule(batch_size, sequence_file, tokenizer, val_split, max_positional_embedding_size, cache, use_weighted_sampler, num_embd, device, validation_chromosome)
    data_module.prepare_data()  # Ensure data is prepared and tokenizer is loaded
    data_module.setup(stage='fit')  # This now computes and stores vocab_size

    model = PromEVEModel(tokenizer, num_layers, num_embd, n_head, max_positional_embedding_size, lr, embeddings_file, device)
    model.prepare_model(data_module.vocab_size)  # Now initialize the model with vocab_size
    

    wandb_logger = None
    if wandb_on:
        # Initialize W&B only in the master process
        wandb_logger = WandbLogger(project='promEVE')

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_output,  # Directory to save the checkpoints
        filename='model_epoch_{epoch}',  # Filename pattern
        save_top_k=-1,  # Save all epochs
        every_n_epochs=1,  # Save every epoch
        monitor='val_loss',  # Metric to monitor for deciding the best model
        mode='min',  # Save the model with the minimum 'val_loss'
        save_weights_only=False  # Save full model
    )

    # Setup GPU usage
    devices = 1 if gpus == 0 else gpus
    accelerator = 'gpu' if gpus > 0 else 'cpu'
    print(f'Using {gpus} GPUS!')

    torch.set_float32_matmul_precision('medium')

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        devices=devices,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],  # Pass the checkpoint callback,
        strategy=DDPStrategy(find_unused_parameters=False),  # Comment out for CPU testing,
        precision="bf16-mixed" # use of on A100 or H100
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint)
    if wandb_on:
        wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser(description="Train a CTRL model with PyTorch Lightning")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training')
    args = parser.parse_args()
    main(args.config, args.checkpoint)

