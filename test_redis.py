import datetime

import yaml
import logging
import argparse

from colorama import Fore
from datasets import concatenate_datasets, Dataset
from easydict import EasyDict

from utils.data_loader import NerProcessor
from utils.arguments import DataTrainingArguments
from utils.main_utils import extract_dataset_by_random_sampling, uncertainty_sampling_multi_models, \
    match_indices_from_base_dataset, get_existing_file_ids
from utils.trainer import ModelTrainer, calculate_threshold

from utils.database.redis_client import RedisVector, RedisPrompt
from utils.llm.agent import Student, get_similar_dataset, get_example
from utils.utils import setup_logging

logging.basicConfig(level=logging.INFO)

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

    setup_logging(file_name="test wikiann en")
    logging.info(f"Base Arguments: {args}")

    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=1,
                                      data_mode=args.mix_dataset_mode
                                      )

    processor = NerProcessor(data_args, test=500)
    initial_train_dataset, pool_dataset, eval_dataset, test_dataset = processor.get_dataset()
    print(f"Initial Train dataset size: {initial_train_dataset.num_rows}")

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

    start_time = datetime.datetime.now()

    model_trainers = EasyDict(d={model_name.split("-")[0]:
                                     {"trainer": ModelTrainer(model_name_or_path=model_name,
                                                              initial_train_dataset=initial_train_dataset,
                                                              valid_dataset=eval_dataset,
                                                              label_list=label_list,
                                                              data_args=data_args),
                                      "embedding_updates": 0,
                                      "early_stopped": False,
                                      "current_score": 0.0,
                                      "training_count": 0}
                                 for model_name in [args.model_name]})

    db_client = RedisVector(dataset_title_value=args.dataset_name,
                            dataset_lang_value=args.dataset_lang)

    prompt_client = RedisPrompt()
    with open("config.yaml", "r") as f:
        agent_config = EasyDict(yaml.safe_load(f))
    peer1 = Student(config=agent_config, db_client=prompt_client,
                    dataset_name=args.dataset_name, dataset_lang=args.dataset_lang)

    trainer = model_trainers.bert.trainer
    embeddings = trainer.get_embeddings(trainer.copy_of_init_train_dataset)
    db_client.insert_vectors(model_name="bert", ids=initial_train_dataset["id"], vectors=embeddings)

    for num in range(ids_to_find := initial_train_dataset.num_rows):
        data_id = initial_train_dataset[num]["id"]
        request_tokens = initial_train_dataset[num]["tokens"]
        logging.info(f"::::::::{data_id}::::::::")
        logging.info(f"Data id: {data_id} >> {request_tokens}")
        vector = db_client.get_vector("bert", str(data_id))
        if vector is not None:
            similar_example = get_similar_dataset(db_client, initial_train_dataset, "bert", vector, data_id)
            examples = get_example(similar_example, id_to_label)
            if similar_example.num_rows > 0:
                logging.info(f"::::::::{data_id}, example exists::::::::")
            else:
                logging.info(f"{Fore.RED}example doesnt exist{Fore.RESET} >> use basic example")
        else:
            basic_example = initial_train_dataset.select(range(2))
            examples = get_example(basic_example, id_to_label)

        find_saved_in_db = db_client.get_all_key_by_pattern(f"memory:*{data_id}*")
        if find_saved_in_db:
            logging.info(f"::::::::{data_id}, already saved in db::::::::")
        else:
            answers = peer1.talk_to_agent(data_id=data_id,
                                          request_tokens=request_tokens,
                                          similar_example=examples,
                                          get_feedback=True,
                                          save_db=True)

            logging.info(f"::::::::{data_id}, saved in db now ! ::::::::")

    end_time = datetime.datetime.now()
    logging.info(peer1.total_api_cost())
    print(peer1.total_api_cost())

    logging.info(
        f"Starts at {start_time.strftime('%Y-%m-%d %H-%M-%S')} -> Ends at {end_time.strftime('%Y-%m-%d %H-%M-%S')}")
    logging.info(f" 소요 시간: {end_time - start_time}")
