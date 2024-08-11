import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict[str, Any]:
    config = {
        "train_batch_size": int(os.getenv("TRAIN_BATCH_SIZE", 8)),
        "val_batch_size": int(os.getenv("VAL_BATCH_SIZE", 1)),
        "num_epochs": int(os.getenv("NUM_EPOCHS", 20)),
        "lr": float(os.getenv("LR", 0.0001)),
        "seq_len": int(os.getenv("SEQ_LEN", 350)),
        "d_model": int(os.getenv("D_MODEL", 512)),
        "src_lang": os.getenv("SRC_LANG", "en"),
        "tgt_lang": os.getenv("TGT_LANG", "it"),
        "dataset_fraction_used": float(os.getenv("DATASET_FRACTION_USED", 1)),
        "model_folder": os.getenv("MODEL_FOLDER", "model"),
        "model_filename": os.getenv("MODEL_FILENAME", "transformer_model_"),
        "log_folder": os.getenv("LOG_FOLDER", "logs"),
        "preload": os.getenv("PRELOAD"),
        "tokenizer_path": os.getenv("TOKENIZER_PATH", "model/tokenizer_{0}.json"),
        "experiment_name": os.getenv("EXPERIMENT_NAME", "runs/tmodel"),
    }

    return config


def get_weights_file_path(config: dict[str, Any], epoch: str) -> str:
    model_folder = config["model_folder"]
    model_filename = config["model_filename"]

    return str(Path(".") / model_folder / model_filename) + epoch + ".pt"

def get_log_folder(config: dict[str, Any]) -> str:
    log_folder = config["log_folder"]
    # Create the log folder if it does not exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    return str(Path(".") / log_folder)

