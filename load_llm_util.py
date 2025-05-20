from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

default_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Recommended quantization type
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation if supported (Ampere+ GPUs)
    bnb_4bit_use_double_quant=True, # Use nested quantization
)

def load_base_model_for_inference(model_path, device='cuda'):
    print(f"\nLoading Full Fine-Tuned model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or torch.mps.is_available()
            else torch.float16
        ),
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded successfully!")
    return model, tokenizer


def load_fft_model_for_inference(model_path, device='cuda'):
    print(f"\nLoading Full Fine-Tuned model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or torch.mps.is_available()
            else torch.float16
        ),
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded successfully!")
    return model, tokenizer


def load_lora_model_for_inference(base_model_id, adapter_path, device='cuda'):
    
    print(f"\nLoading LoRA-{base_model_id} Fine-Tuned model...")
    
    # Load base model again and apply adapters
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or torch.mps.is_available()
            else torch.float16
        ),
        trust_remote_code=True
    ).to(device)
   
    print(f"Loading adapters from '{adapter_path}'...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload() # Merge for faster inference (optional)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded successfully!")
    return model, tokenizer


def load_qlora_model_for_inference(base_model_id, adapter_path, device='cuda', bnb_config=default_bnb_config):
    
    # if device.type != "cuda":
    #     raise RuntimeError(f"Load Error: NVIDIA GPU required, but found '{device.type}'")
    
    print(f"\nLoading QLoRA-{base_model_id} Fine-Tuned model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else torch.float16,
        trust_remote_code=True
    ).to(device)
    
    print(f"Loading adapters from '{adapter_path}'...")
    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded successfully!")
    return model, tokenizer