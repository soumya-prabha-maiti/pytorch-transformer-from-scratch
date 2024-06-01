import logging
from pytorch_transformer_from_scratch.config import get_config,get_log_folder

log_folder = get_log_folder(get_config())

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s.py - %(funcName)s - Line %(lineno)d : %(message)s"
)

shape_logger = logging.getLogger(__name__ + ".shape_logger")
shape_logger.setLevel(logging.DEBUG)
shape_log_filehandler = logging.FileHandler(f"{log_folder}/shapes.log", mode="w")
shape_log_filehandler.setFormatter(log_formatter)
shape_logger.addHandler(shape_log_filehandler)
