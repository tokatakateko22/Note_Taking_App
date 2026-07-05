import os
import warnings
import logging

# Suppress Hugging Face Hub cache, verbosity and authentication warnings globally
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

# Set Hugging Face logger level to ERROR to suppress download and unauthenticated request warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
