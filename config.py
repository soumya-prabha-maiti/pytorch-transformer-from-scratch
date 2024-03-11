from pathlib import Path


def get_config():
    config = {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 0.0001,
        "seq_len": 350,
        "d_model": 512,
        "src_lang": "en",
        "tgt_lang": "it",

        "model_folder": "model", 
        "model_filename": "transformer_model_",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }
    return config

def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = config["model_filename"]

    return str(Path('.')/model_folder/model_filename)