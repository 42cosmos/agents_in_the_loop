import os
import glob
import json
import argparse

from datasets import load_dataset, Dataset

import torch
import evaluate
from seqeval.metrics import classification_report

from utils import check_ratio_of_changed


def add_id(example, idx):
    example['id'] = idx
    return example


def read_yaml(x):
    from easydict import EasyDict
    import yaml
    with open(x, "r") as f:
        return EasyDict(yaml.safe_load(f))


def read_json(x):
    assert os.path.splitext(x)[-1] == ".json"
    with open(x, "r") as f:
        data = json.load(f)
    if "random_ner_tags" not in data.keys():
        if "ner_tags" in data.keys():
            data['random_ner_tags'] = data.pop('ner_tags')
        else:
            return None
    return data


def read_json_list_to_dataset(x: list, return_with_id_list=False):
    assert x, "List is empty ! "
    results = Dataset.from_list([read_json(llm_result) for llm_result in x])
    if return_with_id_list:
        return results, [result["id"] for result in results]
    return results


def match_id_from_base_dataset(base_dataset, result_dataset):
    index_type = type(base_dataset["id"][0])
    indices_to_find = result_dataset["id"]

    if not isinstance(indices_to_find[0], index_type):
        indices_to_find = [index_type(idx) for idx in indices_to_find]
    base_dict = {example['id']: example["ner_tags"] for example in base_dataset}

    filtered_result_dataset = result_dataset.map(lambda example: {'labels': base_dict.get(example['id'], None)})
    assert len(filtered_result_dataset["labels"]) == len(result_dataset["random_ner_tags"])

    return filtered_result_dataset


def compute_metrics(predictions, labels, id_to_label, report=False):
    metric = evaluate.load("seqeval")

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions,
                             references=true_labels,
                             zero_division=0)

    if report:
        print(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_alias", type=str, default="wiki", help="choose one of wnut, conll, wiki, btc")
    parser.add_argument("--data_mode", type=str, default="unlabelled")
    parser.add_argument("--portion", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=256)
    args = parser.parse_args()

    config = read_yaml("/tmp/pycharm_project_960/config.yaml")
    selected_dataset_cfg = config.datasets[args.dataset_name_alias]

    cached_file_name = f"cached-{args.dataset_name_alias}_{args.data_mode}-seq_len_{args.max_seq_length}-10%"
    if "random" in args.data_mode:
        cached_file_name += f"-portion_{args.portion}"

    data_dir = "/home/eunbinpark/workspace/LLM-In-The-Loop/cached_dataset"
    cached_features_file = os.path.join(data_dir, cached_file_name)
    check_cached_file_existed = os.path.exists(cached_features_file)

    loaded_data = torch.load(cached_features_file)
    _, random_pool_dataset, _, _ = loaded_data

    # 데이터 변경 비율 확인
    if args.data_mode == "unlabelled" and args.portion == 1.0:
        from itertools import chain

        check_unlabelled = list(chain(*random_pool_dataset["random_ner_tags"]))
        assert len(set(check_unlabelled)) == 1, f"Unlabelled dataset has more than one label: {set(check_unlabelled)}"
        print(f"All labels are removed from the dataset.")

    llm_result_files = glob.glob(
        f"{data_dir}/LLM/{args.dataset_name_alias}/*.json")

    llm_files_to_dataset = read_json_list_to_dataset(llm_result_files)
    if args.dataset_name_alias == "wiki":
        original_dataset = load_dataset(config.datasets[args.dataset_name_alias].name, "en", split="train")
    else:
        original_dataset = load_dataset(config.datasets[args.dataset_name_alias].name, split="train")

    if "tags" in original_dataset.features:
        original_dataset = original_dataset.rename_column("tags", "ner_tags")

    original_dataset = original_dataset if "id" in original_dataset.features else original_dataset.map(add_id,
                                                                                                       with_indices=True)

    label_list = original_dataset.features["ner_tags"].feature.names
    id_to_label = dict(enumerate(label_list))
    label_to_id = {v: k for k, v in id_to_label.items()}

    llm_dataset = match_id_from_base_dataset(original_dataset, llm_files_to_dataset)

    print(f"LLM Results")
    print(
        compute_metrics(predictions=llm_dataset["random_ner_tags"],
                        labels=llm_dataset["labels"],
                        id_to_label=id_to_label,
                        report=True)
    )