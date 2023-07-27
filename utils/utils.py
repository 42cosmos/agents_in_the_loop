import os
import sys
import json
import logging
import logging.config
import datetime
import time
import traceback

import torch
import nvidia_smi
import subprocess

import yaml
from easydict import EasyDict


def setup_logging(file_name=None):
    main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    proj_dir = os.path.abspath(os.path.join(main_script_dir, ".."))
    logs_dir = os.path.join(main_script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    today = f'[{datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")}]'

    log_config_file = os.path.join(main_script_dir, "logging_config.json")

    if file_name:
        log_file = os.path.join(logs_dir, f"{today}-{file_name}.log")
    else:
        log_file = os.path.join(logs_dir, f"{today}.log")

    os.makedirs(logs_dir, exist_ok=True)

    if os.path.exists(log_config_file):
        with open(log_config_file, 'rt') as f:
            config = json.load(f)
            config["handlers"]["file"].update({"filename": log_file})
            logging.config.dictConfig(config)
    else:
        logging.info("Warning: logging_config.json not found. Using default logging configuration.")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )


def check_gpu_memory():
    nvidia_smi.nvmlInit()
    device_count = torch.cuda.device_count()
    memory_info = {}
    for device_id in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        device_info = {
            "device_name": torch.cuda.get_device_name(device_id),
            "total_memory_MiB": info.total / 1024 ** 2,
            "used_memory_MiB": info.used / 1024 ** 2,
            "free_memory_MiB": info.free / 1024 ** 2,
        }
        memory_info[f"Device {device_id}"] = device_info

    return memory_info


def kill_python_process(conda_env_name: str = "llm"):
    command = f"nvidia-smi | grep '{conda_env_name}/bin/python' | awk '{{ print $5 }}' | xargs -n1 kill -9"
    subprocess.run(command, shell=True)


def read_json(file_path_with_name):
    with open(file_path_with_name, "r") as f:
        return json.load(f)


def read_yaml(file_path_with_name):
    with open(file_path_with_name, "r") as f:
        saved_config = yaml.safe_load(f)
        easy_dict_config = EasyDict(saved_config)
    return easy_dict_config


def update_json(new_data: dict,
                dataset_name: str,
                file_path="/home/eunbinpark/workspace/LLM-In-The-Loop/cached_dataset/LLM",
                required_keys=None):
    if required_keys is None:
        required_keys = ["id", "tokens"]
    assert all(True if key in new_data.keys() else False for key in required_keys)

    file_name = f"{dataset_name}-{new_data['id']}.json"
    os.makedirs(os.path.join(file_path, dataset_name), exist_ok=True)
    file_path = os.path.join(file_path, dataset_name, file_name)

    if file_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, dataset_name, file_name)

    with open(file_path, "w") as f:
        json.dump(new_data, f, ensure_ascii=False)


def handle_general_error(e, logger):
    # 일반적인 오류 처리 부분
    error_msg = traceback.format_exc()
    logger.error(f"{handle_general_error.__name__}: Failed to process future with exception: {e}")
    logger.error(error_msg)


def handle_rate_limit_error(rate_limit_err, logger):
    # RateLimitError 처리 부분
    logger.error(f"{handle_rate_limit_error.__name__}: Rate Limit Error Occurred: {rate_limit_err}")
    time.sleep(60)



