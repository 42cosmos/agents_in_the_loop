import argparse

import yaml
from easydict import EasyDict

from utils.database.redis_client import RedisVector

from utils.data_loader import NerProcessor
from utils.arguments import DataTrainingArguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikiann", choices=["wikiann", "polyglot"])
    parser.add_argument("--dataset_lang", type=str, default="en", choices=["ko", "en", "ja", "pl", "id"])
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument("--mix_dataset_mode", type=str, default="original",
                        choices=["original", "random_entity", "random_word_and_entity", "random_entity_partial",
                                 "unlabelled"])
    parser.add_argument("--portion", type=float, default=1.0)
    args = parser.parse_args()

    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=1,
                                      data_mode=args.mix_dataset_mode
                                      )

    processor = NerProcessor(data_args)
    initial_train_dataset, pool_dataset, eval_dataset, test_dataset = processor.get_dataset()
    print(f"Initial Train dataset size: {initial_train_dataset.num_rows}")

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels
    db_client = RedisVector(dataset_title_value=args.dataset_name,
                            dataset_lang_value=args.dataset_lang,
                            model_title_value=args.model_name)

