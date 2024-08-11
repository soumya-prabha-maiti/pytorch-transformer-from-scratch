import inspect
import json
import logging

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


def log_types(func):
    arg_names = inspect.getfullargspec(func).args

    def wrapper(*args, **kwargs):
        class_name = func.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0]

        func_name = func.__name__

        args_types = {name: type(arg) for name, arg in zip(arg_names, args)}
        kwargs_types = {k: type(v) for k, v in kwargs.items()}
        all_input_types = {**args_types, **kwargs_types}

        for key, val in all_input_types.items():
            type_logger.debug(f"{class_name} :: {func_name} :: input.{key} :: {val}")

        result = func(*args, **kwargs)

        type_logger.debug(f"{class_name} :: {func_name} :: output :: {type(result)}")
        return result

    return wrapper


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