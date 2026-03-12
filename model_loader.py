import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float16,
        device_map = "auto"
    )

    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha = 16,
        target_modules = ["q_proj", "v-proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer