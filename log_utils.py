import logging
import json
from pytorch_transformer_from_scratch.config import get_config, get_log_folder


class JsonFileHandler(logging.Handler):
    def __init__(self, json_file_path):
        super().__init__()
        self.json_file_path = json_file_path
        self.data = {}

    def emit(self, record):
        try:
            # Format the log record
            log_message = self.format(record)

            # Split the message at double colons
            # The record after formatting is expected to be in the format:
            #       timestamp::log_level::module_name::func_name::line_num::message
            # where message is in the format:
            #       nn_block::variable_name::data  (data can be shape, type etc)
            (
                timestamp,
                log_level,
                module_name,
                func_name,
                line_num,
                nn_block,
                variable_name,
                data,
            ) = [item.strip() for item in log_message.split("::")]

            self.data.setdefault(module_name, {})
            self.data[module_name].setdefault(func_name, {})
            self.data[module_name][func_name].setdefault(nn_block, {})
            self.data[module_name][func_name][nn_block][variable_name] = data

            with open(self.json_file_path, "w") as json_file:
                json.dump(self.data, json_file, indent=4)
        except Exception as e:
            self.handleError(record)


log_folder = get_log_folder(get_config())

log_formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(module)s.py :: %(funcName)s :: Line %(lineno)d :: %(message)s"
)

shape_logger = logging.getLogger(__name__ + ".shape_logger")
shape_logger.setLevel(logging.DEBUG)
shape_log_filehandler = logging.FileHandler(f"{log_folder}/shapes.log", mode="w")
shape_log_filehandler.setFormatter(log_formatter)
shape_logger.addHandler(shape_log_filehandler)
json_handler = JsonFileHandler(f"{log_folder}/shapes.json")
json_handler.setFormatter(log_formatter)
shape_logger.addHandler(json_handler)

type_logger = logging.getLogger(__name__ + ".type_logger")
type_logger.setLevel(logging.DEBUG)
type_log_filehandler = logging.FileHandler(f"{log_folder}/types.log", mode="w")
type_log_filehandler.setFormatter(log_formatter)
type_logger.addHandler(type_log_filehandler)