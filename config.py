MODEL_NAME = "google/gemma-2b"

DATA_PATH = "data/all_recipes_combined.json"

OUTPUT_DIR = "./gemma_recipes"
LOG_DIR = "./logs"

MAX_LENGTH = 512

BATCH_SIZE = 1
GRAD_ACCUM = 16

EPOCHS = 3
LR = 2e-4