from typing import Tuple, List, Any, Optional

import torch
from datasets import Dataset

from utils.trainer import communicate_models_for_uncertainty, get_original_labels


def uncertainty_sampling_multi_models(trainers: dict,
                                      id_to_label: dict,
                                      dataset: Dataset,
                                      training_threshold: float) -> tuple[list[Any], float, Optional[Dataset]]:
    model_predictions = []
    for model_name, trainer_dict in trainers.items():
        trainer = trainer_dict["trainer"]
        with torch.no_grad():
            prediction_output = trainer.get_predictions(dataset)
            model_predictions.append(prediction_output)

    uncertainty_ids, average_disagreement_rate, certainty_id_dict = \
        communicate_models_for_uncertainty(*model_predictions,
                                           id_to_label=id_to_label,
                                           threshold=training_threshold)

    certainty_dataset = None
    if certainty_id_dict:
        first_tokenizer = trainers[list(trainers.keys())[0]].trainer.tokenizer
        certainty_datum = []
        for key, predicts in certainty_id_dict.items():
            predict_labels = get_original_labels(**predicts, id_to_label=id_to_label, tokenizer=first_tokenizer)
            if len(predicts["tokens"]) == len(predict_labels):
                certainty_datum.append({"id": key, "tokens": predicts["tokens"], "ner_tags": predict_labels})

        certainty_dataset = Dataset.from_list(certainty_datum, features=dataset.features)

    # DESCRIPTION: average_disagreement_rate: 평균적으로 첫 번째 모델의 예측에 비해 다른 모델들이 average_disagreement_rate% 불일치한다는 것
    return uncertainty_ids, average_disagreement_rate, certainty_dataset


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


def extract_dataset_by_random_sampling(base_dataset, n_samples=100, return_only_indices=False):
    """
    base_dataset에서 n_samples개만큼 랜덤 추출
    :param base_dataset: 오라벨 데이터셋 전체
    :param n_samples: 오라벨 데이터셋에서 랜덤으로 추출할 데이터 개수
    :param return_only_indices: 데이터셋은 제외하고 인덱스만 추출하려고 할 시 TRUE
    :return:
    """
    import numpy as np
    sample_indices = np.random.choice(base_dataset["id"], n_samples, replace=False)
    if return_only_indices:
        return sample_indices

    random_dataset = match_indices_from_base_dataset(base_dataset, sample_indices, remove=False)
    return sample_indices, random_dataset
