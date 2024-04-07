import logging

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s.py - %(funcName)s - Line %(lineno)d : %(message)s"
)

shape_logger = logging.getLogger(__name__ + ".shape_logger")
shape_logger.setLevel(logging.DEBUG)
shape_log_filehandler = logging.FileHandler("shapes.log", mode="w")
shape_log_filehandler.setFormatter(log_formatter)
shape_logger.addHandler(shape_log_filehandler)
