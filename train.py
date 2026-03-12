from transformers import Trainer, TrainingArguments, TrainerCallback
from config import *
from dataset_loader import load_dataset
from formatter import format_recipe
from model_loader import load_model
from tokenizer_utils import tokenize_function
from tqdm import tqdm

class TQDMProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # log every 10 steps
            print(f"Step {state.global_step} - loss: {state.log_history[-1]['loss']:.4f}")

def train(resume_checkpoint = None):

    model, tokenizer = load_model()

    train_dataset, val_dataset = load_dataset()

    train_dataset = train_dataset.map(format_recipe)
    val_dataset = val_dataset.map(format_recipe)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True,
        remove_columns=train_dataset.column_names
        )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True,
        remove_columns=val_dataset.column_names
        )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_dir=LOG_DIR,
        logging_steps=10,
        logging_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[TQDMProgressCallback]
    )

    print("\nStarting training...\n")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("\nTraining completed.\n")

    model.save_pretrained("italian_recipes_gemma")
    tokenizer.save_pretrained("italian_recipes_gemma")