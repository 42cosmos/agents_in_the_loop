import os
import random
import logging
from functools import partial

import numpy as np

import torch
from datasets import load_dataset


class NerProcessor:
    def __init__(self, args):
        self.logger = logging.getLogger(__name__.split(".")[-1])

        self.args = args
        self.mode = args.data_mode

        cached_file_name = f"cached-{self.args.dataset_name}_{self.args.dataset_lang}_{self.args.data_mode}-seq_len_{self.args.max_seq_length}"
        if "random" in self.args.data_mode:
            cached_file_name += f"-portion_{self.args.portion}"

        if "original" != self.args.data_mode:
            cached_file_name += f"-{self.args.initial_train_n_percentage}%"

        self.cached_file_name = cached_file_name
        self.cached_features_file = os.path.join(self.args.data_dir, cached_file_name)
        self.existed_cached_file = os.path.exists(self.cached_features_file)

    def get_cached_dataset(self):
        if self.existed_cached_file:
            self.logger.info(f"Loading features from cached file {self.cached_file_name}")
            loaded_data = torch.load(self.cached_features_file)
            if self.args.make_data_pool:
                self.initial_train_dataset, self.pool_dataset, self.valid_dataset, self.test_dataset = loaded_data
                self.update_label_list()
                return self.initial_train_dataset, self.pool_dataset, self.valid_dataset, self.test_dataset
            else:
                self.train_dataset, self.valid_dataset, self.test_dataset = loaded_data
                self.update_label_list()
                return self.train_dataset, self.valid_dataset

    def update_label_list(self):
        self.labels: list = self.valid_dataset.features["ner_tags"].feature.names
        assert self.labels, f"Labels are empty. Check the label_column_name '{self.args.label_column_name}'"
        self.label_to_id, self.id_to_label = self.get_label_map(self.labels)
        self.ignore_entity = self.label_to_id["O"]
        self.start_tag = min(key for key in self.id_to_label if key != self.ignore_entity)
        self.end_tag = max(key for key in self.id_to_label if key != self.ignore_entity)

    def get_dataset(self):
        cached_file = self.get_cached_dataset()
        if cached_file:
            return cached_file

        self.logger.info(f"Creating dataset named '{self.args.dataset_name}'")
        if self.args.dataset_name == "wikiann":
            self.raw_dataset = load_dataset(self.args.dataset_name, self.args.dataset_lang)
        else:
            self.raw_dataset = load_dataset(self.args.dataset_name)

        self.raw_dataset = self.raw_dataset if "id" in self.raw_dataset["train"].features else \
            self.raw_dataset.map(self.add_id, with_indices=True)

        required_columns = ['id', 'tokens', 'ner_tags']
        for split in self.raw_dataset.keys():
            self.raw_dataset[split] = self.raw_dataset[split].remove_columns(
                [col for col in self.raw_dataset[split].column_names if col not in required_columns])

        self.train_dataset = self.raw_dataset["train"]
        self.valid_dataset = self.raw_dataset["validation"]
        self.test_dataset = self.raw_dataset["test"] if "test" in self.raw_dataset else None

        # label list 설정
        self.update_label_list()

        if self.args.make_data_pool:
            # Split dataset into initial train, pool
            initial_train_samples = int((self.args.initial_train_n_percentage * self.train_dataset.num_rows) / 100)
            self.logger.info(f"Split initial train dataset -> {initial_train_samples} samples")

            train_indices = np.random.choice(self.train_dataset.num_rows, initial_train_samples, replace=False)
            pool_indices = np.setdiff1d(np.arange(self.train_dataset.num_rows), train_indices)

            # train_dataset은 실 답, pool_dataset은 에러가 섞인 값
            initial_train_dataset = self.train_dataset.select(train_indices)
            if self.mode != "original":
                initial_train_dataset = initial_train_dataset.rename_column("ner_tags", "random_ner_tags")

            pool_dataset = self.train_dataset.select(pool_indices)
            pool_dataset = self.apply_mode(pool_dataset)

            self.logger.info(f"Save features to cached file {self.cached_file_name}")
            torch.save((initial_train_dataset, pool_dataset, self.valid_dataset, self.test_dataset),
                       self.cached_features_file)

            return initial_train_dataset, pool_dataset, self.valid_dataset, self.test_dataset

        else:
            self.logger.info(f"Save features to cached file {self.cached_file_name}")
            self.train_dataset = self.apply_mode(self.train_dataset)
            torch.save((self.train_dataset, self.valid_dataset, self.test_dataset), self.cached_features_file)
            return self.train_dataset, self.valid_dataset, self.test_dataset

    def apply_mode(self, dataset):
        if self.mode == "random_entity":
            dataset = dataset.map(self.make_random_entity)

        elif self.mode == "unlabelled":
            dataset = dataset.map(self.make_unlabelled)

        elif self.mode == "random_word_and_entity":
            dataset = dataset.map(self.make_random_word_and_entity)

        elif self.mode == "random_entity_partial":
            # 전체 엔티티 중 B-XXX만 카운트
            # b_values = [v for i, v in self.label_to_id.items() if i.startswith("B-")]
            counted_all_entity = sum(
                [1 for tags in dataset["ner_tags"] for tag in tags if tag != self.ignore_entity])

            # 변경해야 할 엔티티 수
            self.counted_target_entity = round(counted_all_entity * self.args.portion)
            self.logger.info(f"전체 엔티티 수: {counted_all_entity} | 변경해야 할 엔티티 수: {self.counted_target_entity}")

            # 한 문장 당 몇 개를 변경할 건가?
            # num_changes_per_sent = counted_target_entity // dataset.num_rows
            # assert num_changes_per_sent == 0, "num_changes_per_sent should be greater than 0"
            # self.logger.info(
            #     f"전체 엔티티 수: {counted_all_entity} | 변경해야 할 엔티티 수: {counted_target_entity} -> {num_changes_per_sent}")

            return make_random_partial_entity(dataset, self.counted_target_entity)
        return dataset

    def make_unlabelled(self, example):
        ner_tags = example["ner_tags"]
        no_entity = [self.ignore_entity for _ in ner_tags]
        example["random_ner_tags"] = no_entity

        return example

    def make_random_entity(self, example):
        """
        Randomly replace the entity with another entity
        word는 제대로 태깅되었다고 가정
        """
        ner_tags = example["ner_tags"]

        random_entity = [
            random.randint(self.start_tag, self.end_tag + 1) if tag != self.ignore_entity else self.ignore_entity
            for tag
            in ner_tags]
        example["random_ner_tags"] = random_entity

        return example

    def make_random_word_and_entity(self, example):
        """
        Randomly replace the word and entity with another word and entity
        word, entity 모두 제대로 태깅되지 않았다는 가정
        """
        ner_tags = example["ner_tags"]
        # 0이상 len(labels) - 1 이하
        random_tags = [random.randint(a=0, b=len(self.labels) - 1) for _ in range(len(ner_tags))]
        example["random_ner_tags"] = random_tags

        return example

    @staticmethod
    def get_label_map(labels: list):
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = dict(enumerate(labels))
        return label_to_id, id_to_label

    @staticmethod
    def add_id(example, idx):
        example['id'] = idx
        return example


def make_random_partial_entity(dataset, counted_target_entity):
    labels = dataset.features["ner_tags"].feature.names
    id_to_label = dict(enumerate(labels))
    label_to_id = {v: k for k, v in id_to_label.items()}
    ignore_entity = label_to_id["O"]
    start_tag = min(key for key in id_to_label if key != ignore_entity)
    end_tag = max(key for key in id_to_label if key != ignore_entity)

    mixed_entity = dataset["ner_tags"].copy()

    for entity_idx, entities in enumerate(mixed_entity):
        if counted_target_entity <= 0:
            continue
        for idx, tag in enumerate(entities):
            if tag == ignore_entity:
                mixed_entity[entity_idx][idx] = tag
                continue

            if counted_target_entity <= 0:
                mixed_entity[entity_idx][idx] = tag
                continue

            entity = id_to_label[tag].split("-")[1]
            avoid_same_entity = [v for k, v in label_to_id.items() if entity in k]
            possible_values = [i for i in range(start_tag, end_tag + 1) if i != tag and i not in avoid_same_entity]
            mixed_entity[entity_idx][idx] = random.choice(possible_values)
            counted_target_entity -= 1

    return dataset.add_column("random_ner_tags", mixed_entity)


def check_ratio_of_changed_entity(dataset):
    from itertools import chain
    from collections import Counter

    labels = dataset.features["ner_tags"].feature.names
    id_to_label = dict(enumerate(labels))

    random_entity_flat = list(chain.from_iterable(dataset["random_ner_tags"]))
    original_flat = list(chain.from_iterable(dataset["ner_tags"]))

    # Count the frequency of each number
    random_counter = Counter(random_entity_flat)
    original_counter = Counter(original_flat)

    # Calculate the ratio
    ratios = {}
    for k in range(0, len(labels) - 1):
        random_count = random_counter[k]
        original_count = original_counter[k]
        ratio = random_count / original_count if original_count != 0 else np.inf
        ratios[id_to_label[k]] = ratio

    return ratios


def check_ratio_of_changed(dataset, mode="random_word_and_entity"):
    from itertools import chain
    labels = dataset.features["ner_tags"].feature.names
    id_to_label = dict(enumerate(labels))
    label_to_id = {v: k for k, v in id_to_label.items()}
    ignore_entity = label_to_id["O"]

    random_entity = list(chain.from_iterable(dataset["random_ner_tags"]))
    original = list(chain.from_iterable(dataset["ner_tags"]))

    random_entity_flatten = np.array(random_entity).flatten()
    original_flatten = np.array(original).flatten()

    if mode == "random_word_and_entity":
        random_entity_flatten = np.array(random_entity).flatten()
        original_flatten = np.array(original).flatten()

        # Compare only the non-"O" entity elements
        changed_elements = np.count_nonzero(random_entity_flatten != original_flatten)

        # Calculate the total number of non-"O" entity elements
        total_elements = np.count_nonzero(random_entity_flatten)

        # Calculate the percentage of changed non-"O" entity elements
        changed_percentage = (changed_elements / total_elements) * 100
        return changed_percentage

    # Create boolean masks for non-"O" entity elements
    random_entity_mask = random_entity_flatten != ignore_entity
    original_mask = original_flatten != ignore_entity

    # Compare only the non-"O" entity elements
    changed_elements = np.count_nonzero(random_entity_flatten[random_entity_mask] != original_flatten[original_mask])

    # Calculate the total number of non-"O" entity elements
    total_elements = np.count_nonzero(random_entity_mask)

    # Calculate the percentage of changed non-"O" entity elements
    changed_percentage = (changed_elements / total_elements) * 100

    return changed_percentage


def find_unique_elements(list_1, list_2):
    list_1_set = set(list_1)
    list_2_set = set(list_2)

    unique_to_list_1 = list_1_set - list_2_set
    unique_to_list_2 = list_2_set - list_1_set

    return unique_to_list_1 if unique_to_list_1 else unique_to_list_2
