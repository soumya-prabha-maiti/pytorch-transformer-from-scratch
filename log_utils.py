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
            #       timestamp::log_level::message
            # where message is in the format:
            #       class_name(nn_block)::function_name::variable_name::data  (data can be shape, type etc)
            (
                timestamp,
                log_level,
                class_name,
                func_name,
                variable_name,
                data,
            ) = [item.strip() for item in log_message.split("::")]

            self.data.setdefault(class_name, {})
            self.data[class_name].setdefault(func_name, {})
            self.data[class_name][func_name][variable_name] = data

            with open(self.json_file_path, "w") as json_file:
                json.dump(self.data, json_file, indent=4)
        except Exception as e:
            self.handleError(record)


def log_metadata(log_shapes=True, log_types=True):
    def log_metadata_without_args(func):
        arg_names = inspect.getfullargspec(func).args

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            class_name = func.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0]
            func_name = func.__name__
            args_dict = {name: arg for name, arg in zip(arg_names, args)}

            all_input_types = {}
            all_input_shapes = {}
            for name, arg in {**args_dict, **kwargs}.items():
                all_input_types[f"{name}"] = type(arg)
                try:
                    all_input_shapes[f"{name}.shape"] = arg.shape
                except:
                    try:
                        all_input_shapes[f"{name}.__len__()"] = len(arg)
                    except:
                        all_input_shapes[f"{name}"] = "shape or len not available"

            if log_types:
                for name, val in all_input_types.items():
                    type_logger.debug(f"{class_name} :: {func_name} :: input.{name} :: {val}")
                type_logger.debug(f"{class_name} :: {func_name} :: output :: {type(result)}")

            if log_shapes:
                for name, val in all_input_shapes.items():
                    shape_logger.debug(f"{class_name} :: {func_name} :: input.{name} :: {val}")
                try:
                    shape_logger.debug(f"{class_name} :: {func_name} :: output.shape :: {result.shape}")
                except:
                    try:
                        shape_logger.debug(f"{class_name} :: {func_name} :: output.__len__() :: {len(result)}")
                    except:
                        shape_logger.debug(f"{class_name} :: {func_name} :: output :: shape or len not available")

            return result

        return wrapper
    
    return log_metadata_without_args



log_folder = get_log_folder(get_config())

log_formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")

shape_logger = logging.getLogger(__name__ + ".shape_logger")
shape_logger.setLevel(logging.DEBUG)
shape_log_filehandler = logging.FileHandler(f"{log_folder}/shapes.log", mode="w")
shape_log_filehandler.setFormatter(log_formatter)
shape_logger.addHandler(shape_log_filehandler)
shape_json_filehandler = JsonFileHandler(f"{log_folder}/shapes.json")
shape_json_filehandler.setFormatter(log_formatter)
shape_logger.addHandler(shape_json_filehandler)

type_logger = logging.getLogger(__name__ + ".type_logger")
type_logger.setLevel(logging.DEBUG)
type_log_filehandler = logging.FileHandler(f"{log_folder}/types.log", mode="w")
type_log_filehandler.setFormatter(log_formatter)
type_logger.addHandler(type_log_filehandler)
type_json_filehandler = JsonFileHandler(f"{log_folder}/types.json")
type_json_filehandler.setFormatter(log_formatter)
type_logger.addHandler(type_json_filehandler)
