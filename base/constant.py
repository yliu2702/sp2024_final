import os

BASE_DIR = "/projectnb/ds598/projects/yliu2702/"

DATA_BASE_DIR = "/projectnb/ds598/projects/yliu2702/data"

# Set the Transformers cache directory
os.environ["TRANSFORMERS_CACHE"] = BASE_DIR + "misc"

# Set the Hugging Face home directory (this includes datasets cache)
os.environ["HF_HOME"] = BASE_DIR + "misc"
os.environ["TORCH_HOME"] = BASE_DIR + "misc"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SAVE PATHS
DEMO_SAVE_PATH = BASE_DIR + "RESULTS/bert-base"
