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

logging.basicConfig(level=logging.INFO)


def get_llm_labeling(agent,
                     db_conn,
                     dataset,
                     indices,
                     model_name,
                     label_to_id,
                     dataset_name,
                     data_dir,
                     find_saved_file=True):
    # labelling 할 대상
    need_to_be_found_dataset = match_indices_from_base_dataset(base_dataset=dataset,
                                                               indices_to_find=indices, remove=False)

    # labelling 할 대상 중에서 기존에 labelling 된 데이터를 제외
    if find_saved_file:
        # db_conn에 id 리스트를 넘겨주어 값이 있나 확인 ( key 값만 확인하면 될듯 ! )
        pass

    find_id_count: int = need_to_be_found_dataset.num_rows
    for num in range(find_id_count):
        data_id = initial_train_dataset[num]["id"]
        vector = db_client.get_vector(model_name, data_id)
        assert vector is not None, "vector is None!!"

        similar_example = get_similar_dataset(db_client, initial_train_dataset, model_name, vector, data_id)
        examples = get_example(similar_example, id_to_label)
        if similar_example.num_rows > 0:
            logging.info(f"{data_id}, example exists")
        else:
            raise f"{Fore.RED}example doesnt exist{Fore.RESET}"

        answers = agent.talk_to_agent(data_id=initial_train_dataset[num]["id"],
                                      request_tokens=initial_train_dataset[num]["tokens"],
                                      similar_example=examples,
                                      get_feedback=True,
                                      save_db=False)
        return answers


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

    processor = NerProcessor(data_args, test=100)
    initial_train_dataset, pool_dataset, eval_dataset, test_dataset = processor.get_dataset()
    print(f"Initial Train dataset size: {initial_train_dataset.num_rows}")

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

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

    current_score = 0
    target_score = 0.9
    tokens_used = 0

    start_threshold = 0.1
    current_step = 0
    total_steps = 10
    start_time = datetime.datetime.today()

    while target_score > current_score:
        current_threshold = calculate_threshold(start=start_threshold,
                                                end=target_score,
                                                current_step=current_step,
                                                total_steps=total_steps)

        if all(trainer_dict["early_stopped"] for trainer_dict in model_trainers.values()):
            logging.warning(f"All ModelTrainer instances are early stopped. ")
            break

        for model_name, trainer_dict in model_trainers.items():
            trainer = trainer_dict["trainer"]

            embeddings_of_initial_dataset = \
                trainer.get_embeddings(trainer.copy_of_init_train_dataset)

            # 최초 데이터셋 임베딩 지속적 업데이트
            db_client.insert_vectors(model_name=model_name,
                                     ids=initial_train_dataset["id"],
                                     vectors=embeddings_of_initial_dataset)

            if trainer_dict["early_stopped"]:
                continue

            if trainer.early_stopping_callback.early_stopped:
                trainer_dict["early_stopped"] = True
                logging.info(f">>>>> {model_name} is early stopped <<<<< ")

            trainer.train()
            trainer_dict["training_count"] += 1

            # Reset the learning rate
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = 2e-5

            eval_result = trainer.evaluate()
            current_score = round(eval_result["eval_f1"], 4)
            trainer_dict["current_score"] = current_score

            logging.info(f"***** Evaluation metrics for {model_name} *****")
            for k, v in eval_result.items():
                logging.info(f"  {k} = {v}")

            logging.info(
                f"Current F1 for {model_name}: {current_score} // Training count: {trainer_dict['training_count']}")

        # 풀 데이터셋에서 n samples 수 만큼 랜덤하게 데이터를 추출: untokenize 된 데이터셋
        _, sample_pool_dataset = extract_dataset_by_random_sampling(dataset=pool_dataset,
                                                                    n_samples=data_args.n_samples,
                                                                    return_only_indices=False)

        # Uncertainty sampling
        # 모든 모델이 동의하지 않으면 Uncertainty 라고 판단하며, 각 예측에 대해 모델들 간의 불일치 비율을 계산
        uncertain_indices, disagreement_rate = uncertainty_sampling_multi_models(trainers=model_trainers,
                                                                                 dataset=sample_pool_dataset,
                                                                                 id_to_label=id_to_label,
                                                                                 training_threshold=current_threshold)

        logging.info(f"Current Threshold: {current_threshold}")
        logging.info(f"Number of Incomplete Data: {len(uncertain_indices)}")
        logging.info(f"Percentage of disagreement between models for each prediction: {disagreement_rate}")

        if not uncertain_indices:
            logging.info(f"No uncertain indices found.")
            new_label_dataset = sample_pool_dataset
        else:
            if args.ask_oracle:
                new_label_dataset = get_llm_labeling(agent=peer1,
                                                     model_name="?",
                                                     indices=uncertain_indices,
                                                     dataset=sample_pool_dataset,
                                                     data_dir=data_args.data_dir,
                                                     label_to_id=label_to_id,
                                                     dataset_name=data_args.dataset_name,
                                                     find_saved_file=True)

                tokens_used += peer1.total_api_cost()

                # 기존 pool dataset 에서 uncertain_indices 제거
                pool_dataset = match_indices_from_base_dataset(base_dataset=pool_dataset,
                                                               indices_to_find=uncertain_indices,
                                                               remove=True)

                # uncertain_indices 를 제거한 pool dataset 에 새로 라벨링된 데이터를 추가
                pool_dataset = concatenate_datasets([pool_dataset, new_label_dataset])

            else:
                new_label_dataset = sample_pool_dataset
        new_label_dataset = new_label_dataset.cast(initial_train_dataset.features)

        for model_name, trainer_dict in model_trainers.items():
            if trainer_dict["early_stopped"]:
                continue
            updated_trainer = trainer_dict["trainer"]
            updated_trainer.update_dataset(new_label_dataset)
            logging.info(
                f"***** Updated training dataset for {model_name} length -> {updated_trainer.train_dataset.num_rows} *****")

        current_step += 1

    end_time = datetime.datetime.today()
    logging.info("=" * 50)
    for k, v in model_trainers.items():
        logging.info(f"***** Final {k} F1 score: {v['current_score']} / Training {v['training_count']} times *****")
        logging.info(f"***** Used Train dataset: {initial_train_dataset.num_rows} *****")

    logging.info(f"***** Token used: {tokens_used}, Price: {tokens_used * 0.0000020} *****")
    logging.info(
        f"Starts at {start_time.strftime('%Y-%m-%d %H-%M-%S')} -> Ends at {end_time.strftime('%Y-%m-%d %H-%M-%S')}")
    logging.info(f" 소요 시간: {end_time - start_time}")

    if args.inference:
        if test_dataset is not None:
            if "tags" in test_dataset.column_names:
                test_dataset = test_dataset.rename_column("tags", "ner_tags")
            if args.mix_dataset_mode != "original":
                test_dataset = test_dataset.rename_column("ner_tags", "random_ner_tags")

            for model_name, trainer_dict in model_trainers.items():
                trainer = trainer_dict["trainer"]
                test_result = trainer.get_predictions(test_dataset)

                logging.info(f"***** Test metrics for {model_name} *****")
                for k, v in test_result.metrics.items():
                    logging.info(f"  {k} = {v}")
                # torch.save(trainer.train_dataset, f"{data_args.dataset_name}-{model_name}-train_dataset")
        else:
            logging.warning(f"Test dataset is not provided.")

    if args.save_model:
        for model_name, trainer_dict in model_trainers.items():
            trainer_dict["trainer"].save_model(
                f"{data_args.dataset_name}-{model_name}-{args.mix_dataset_mode}-ask_oracle_{args.ask_oracle}")
