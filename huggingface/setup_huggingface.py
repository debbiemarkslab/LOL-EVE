#!/usr/bin/env python3
"""
Setup script for LOL-EVE Hugging Face model.
This script helps prepare the model for upload to Hugging Face Hub.
"""

import os
import shutil
from pathlib import Path

def setup_huggingface_model():
    """Setup the Hugging Face model directory with all necessary files."""
    
    print("🧬 Setting up LOL-EVE for Hugging Face Hub")
    print("=" * 50)
    
    # Create the model directory structure
    model_dir = Path("lol-eve-model")
    model_dir.mkdir(exist_ok=True)
    
    # Files to include in the Hugging Face model
    files_to_copy = [
        "README.md",
        "requirements.txt", 
        "example_usage.py",
        "modeling_loleve.py"
    ]
    
    print("Copying essential files...")
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, model_dir / file)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️  {file} not found")
    
    # Create a simple config.json for the model
    config_content = """{
  "architectures": [
    "LOLEVEForCausalLM"
  ],
  "model_type": "loleve",
  "num_layers": 12,
  "num_embd": 768,
  "num_heads": 12,
  "max_positional_embedding_size": 1007,
  "position_embedding_type": "adaptive",
  "use_control_codes": 1,
  "vocab_size": 39378,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "unk_token_id": 3,
  "sep_token_id": 4,
  "mask_token_id": 5,
  "transformers_version": "4.35.0",
  "auto_map": {
    "AutoConfig": "modeling_loleve.LOLEVEConfig",
    "AutoModelForCausalLM": "modeling_loleve.LOLEVEForCausalLM"
  },
  "model_name": "LOL-EVE",
  "description": "Language-Optimized Learning for Evolutionary Variant Effects - A genomic language model for variant effect prediction",
  "task": "text-generation",
  "language": "genomic",
  "license": "mit"
}"""
    
    with open(model_dir / "config.json", "w") as f:
        f.write(config_content)
    print("✅ Created config.json")
    
    print(f"\n📁 Model directory created: {model_dir}")
    print("\nNext steps:")
    print("1. Add your trained model weights to the directory")
    print("2. Add tokenizer files (tokenizer.json, tokenizer_config.json)")
    print("3. Upload to Hugging Face Hub using:")
    print(f"   huggingface-cli upload Marks-lab/LOL-EVE {model_dir}/*")
    
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    setup_huggingface_model()
