from transformers import PreTrainedTokenizerFast
import os 
import argparse 
from tqdm import tqdm 
from datasets import load_dataset 
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq  # Correct import for parquet module
import torch

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")

    args = parser.parse_args()
    base_dir=''
    cache_dir='/cache'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{base_dir}/tokenizer_template',
                                        unk_token="[UNK]",
                                        sep_token="[SEP]",
                                        pad_token="[PAD]",
                                        cls_token="[CLS]")

    dataset = load_dataset("parquet", data_files=args.input, split="train",
                           cache_dir=f"{cache_dir}")
    new_values = set() 
    for row in tqdm(dataset):
        new_values.update(set(row['sequence']))
        new_values.add(row['gene'])
        new_values.add(row['species'])
        new_values.add(row['clade'])
    tokenizer.add_tokens(sorted(list(new_values)))
    tokenizer.add_tokens('start')
    tokenizer.add_tokens('end')
    tokenizer.save_pretrained(args.output)

