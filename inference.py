import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "italian_recipes_gemma"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
print("\nItalian Recipe AI")
print("-----------------")

prompt = input("\nEnter your prompt: ")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Recipe:\n")
print(result)