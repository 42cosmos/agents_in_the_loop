import os
import glob
import logging
from typing import Tuple

import torch
from datasets import Dataset

from utils.utils import read_json
from utils.trainer import communicate_models_for_uncertainty

from utils.llm.token_counter import count_message_tokens


def uncertainty_sampling_multi_models(trainers: dict, dataset, id_to_label, training_threshold) -> Tuple[list, float]:
    model_predictions = []
    for model_name, trainer_dict in trainers.items():
        trainer = trainer_dict["trainer"]
        with torch.no_grad():
            prediction_output = trainer.get_predictions(dataset)
            model_predictions.append(prediction_output)

    uncertainty_ids, average_disagreement_rate = communicate_models_for_uncertainty(*model_predictions,
                                                                                    id_to_label=id_to_label,
                                                                                    threshold=training_threshold)
    # DESCRIPTION: average_disagreement_rate: 평균적으로 첫 번째 모델의 예측에 비해 다른 모델들이 average_disagreement_rate% 불일치한다는 것

    return uncertainty_ids, average_disagreement_rate


def get_existing_file_ids(indices: list, data_dir: str, dataset_name: str, return_abs_path=True):
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    llm_file_dir = glob.glob(os.path.join(dataset_dir, "*.json"))

    results = [id_ for id_ in indices if os.path.join(dataset_dir, f"{dataset_name}-{id_}.json") in llm_file_dir]

    if return_abs_path:
        results = [file_path for id_ in indices
                   if (file_path := os.path.join(dataset_dir, f"{dataset_name}-{id_}.json")) in llm_file_dir]

    if results:
        return results

    return []


def match_indices_from_base_dataset(base_dataset, indices_to_find, remove=True):
    """
    Returns the indices to keep from the full indices list
    :param base_dataset: Huggingface Dataset object with "id" key
    :param indices_to_find: list of indices to remove
    :param remove: bool type to remove or keep the indices
    """
    index_type = type(base_dataset["id"][0])
    if not isinstance(indices_to_find[0], index_type):
        indices_to_find = [index_type(idx) for idx in indices_to_find]

    if remove:
        # indices_to_find 에 있는 id 값만 base_dataset 에서 제거
        filtered_base_dataset = base_dataset.filter(lambda example: example['id'] not in indices_to_find)
    else:
        # indices_to_find 에 있는 id 값만을 추출
        filtered_base_dataset = base_dataset.filter(lambda example: example['id'] in indices_to_find)

    return filtered_base_dataset


def extract_dataset_by_random_sampling(dataset, n_samples=100, return_only_indices=False):
    import numpy as np
    sample_indices = np.random.choice(dataset["id"], n_samples, replace=False)
    if return_only_indices:
        return sample_indices

    random_dataset = match_indices_from_base_dataset(dataset, sample_indices, remove=False)
    return sample_indices, random_dataset


def llm_data_file_tobe_dataset(file_indices: list, data_dir: str, dataset_name: str):
    existed_files_abs_path: list = get_existing_file_ids(indices=file_indices, data_dir=data_dir,
                                                         dataset_name=dataset_name, return_abs_path=True)

    logging.info(
        f"{llm_data_file_tobe_dataset.__name__}: ===== Existed llm results: {len(existed_files_abs_path)} =====")
    all_tokens = 0
    dataset_list = []
    for file_path in existed_files_abs_path:
        data = read_json(file_path)
        if "random_ner_tags" not in data.keys():
            if "ner_tags" in data.keys():
                data['random_ner_tags'] = data.pop('ner_tags')
            else:
                continue
        # TODO: Message로 변경해야함 !
        all_tokens += count_message_tokens(text=str(data["tokens"]), dataset_name=dataset_name)
        dataset_list.append(data)

    if dataset_list:
        return Dataset.from_list(dataset_list), all_tokens
