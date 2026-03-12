from config import MAX_LENGTH

def tokenize_function(examples, tokenizer):

    tokens = tokenizer(
        examples["text"],
        truncation = True,
        padding = "max_length",
        max_length = MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

