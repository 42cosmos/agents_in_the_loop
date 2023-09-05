import os

# Pycharm 외부에서 해당 파일 실행 시 주석 해제 필수 (Pycharm 내부에서 실행 시 주석 처리 필수)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime

import logging
import argparse
from functools import partial
from typing import List

import yaml
from colorama import Fore
from easydict import EasyDict
from tqdm import tqdm
import numpy as np

import torch
from datasets import concatenate_datasets, Dataset

from utils.llm.base import ChatModelResponse, MessageFunctionCall
from utils.utils import (
    check_gpu_memory,
    kill_python_process,
    setup_logging,
    handle_general_error,
)

from utils.main_utils import (
    match_indices_from_base_dataset,
    extract_dataset_by_random_sampling, uncertainty_sampling_multi_models
)

from utils.database.redis_client import RedisVector, RedisLLMResponse
from utils.data_loader import NerProcessor
from utils.trainer import (
    calculate_threshold,
    ModelTrainer)

from utils.llm.agent import Student, answer_to_data, get_similar_dataset, get_example

from utils.arguments import DataTrainingArguments

import concurrent.futures


def process_llm(random_data,
                prompt_examples: List[MessageFunctionCall],
                agent: Student,
                db_client: RedisLLMResponse) -> List[ChatModelResponse]:
    logger = logging.getLogger(f"{process_llm.__name__}-{random_data['id']}")
    logger.setLevel(logging.DEBUG)

    all_agents_answer = agent.talk_to_agent(raw_data=random_data,
                                            similar_examples=prompt_examples,
                                            get_feedback=True,
                                            db_prompt_client=db_client)

    return all_agents_answer


def add_new_features(example, model_trainers, db_client, base_dataset):
    from collections import defaultdict
    example_dict = defaultdict(lambda: [])
    example_dataset = Dataset.from_list([example])
    for model_name, trainer_dict in model_trainers.items():
        trainer: ModelTrainer = trainer_dict.trainer
        embedding_value = trainer.get_embeddings(example_dataset)
        similar_example_dataset = get_similar_dataset(db_client=db_client,
                                                      dataset=base_dataset,
                                                      model_name=model_name,
                                                      vector=embedding_value,
                                                      data_id=example["id"],
                                                      num=1)

        similar_example = get_example(similar_example_dataset, id_to_label=trainer.id_to_label)

        if similar_example_dataset.num_rows > 0:
            logging.info(f"{model_name} - Similar example found: {example['id']} -> {similar_example_dataset['id'][0]}")
        else:
            logging.info(f"No similar example found.")
            import random
            random_number = random.randint(0, base_dataset.num_rows)
            basic_example = base_dataset.select([random_number])
            similar_example = get_example(basic_example, trainer.id_to_label)

        example_dict[example["id"]].append(similar_example[0])
    return example_dict


def create_futures(executor,
                   agent: Student,
                   propmt_examples: dict,
                   random_pool_dataset: Dataset,
                   db_client: RedisLLMResponse,
                   ):
    futures = [executor.submit(process_llm, example, propmt_examples[example["id"]], agent, db_client)
               for example in random_pool_dataset]
    return futures


def process_futures(futures, pbar):
    """
    생성된 futures를 처리하고 결과를 수집하는 함수
    :param futures: 멀티스레드의 결과
    :param pbar: tqdm bar
    """
    logger = logging.getLogger(f"{process_futures.__name__}")
    future_results = []
    for future in concurrent.futures.as_completed(futures):
        llm_responses = future.result()
        try:
            future_results.append(llm_responses[-1])
            pbar.update(1)

        except Exception as e:
            handle_general_error(e, logger)
            logger.error(f"Failed to process future with exception: {llm_responses[-1]}")
    return future_results


def find_exist_files_in_db(indices, db_client):
    exist_data_ids = db_client.get_keys_by_ids_with_lua(indices)
    # memory:student:dataset_name:dataset_lang:data_id:conv_num 에서
    # dataset_name:dataset_lang:data_id만 추출
    if exist_data_ids:
        return [":".join(data.split(":")[2:-1]) for data in exist_data_ids]
    # DB에 저장된 파일이 존재하는 경우
    return []


def get_llm_labeling(agent: Student,
                     sample_pool_dataset: Dataset,
                     uncertain_data_indices: List[str],
                     db_client: RedisLLMResponse,
                     label_to_id,
                     num_threads=20
                     ):
    logger = logging.getLogger(f"{get_llm_labeling.__name__}")
    logger.setLevel(logging.DEBUG)

    existed_llm_ids = find_exist_files_in_db(indices=uncertain_data_indices, db_client=db_client)
    indices_requiring_labelling = list(set(uncertain_data_indices) - set(existed_llm_ids))
    logger.info(f"{Fore.YELLOW}Existed Response data length {len(existed_llm_ids)}{Fore.RESET}")
    logger.info(f"{Fore.YELLOW}Required labelling data length {len(indices_requiring_labelling)}{Fore.RESET}")

    if len(indices_requiring_labelling) > 0:
        # 태깅해야 할 인덱스 - 태깅 결과가 저장되어있는 파일 => 태깅해야 할 인덱스
        dataset_requiring_labelling = match_indices_from_base_dataset(base_dataset=sample_pool_dataset,
                                                                      indices_to_find=indices_requiring_labelling,
                                                                      remove=False)

        encode_fn = partial(add_new_features,
                            model_trainers=model_trainers,
                            db_client=db_vector_client,
                            base_dataset=initial_train_dataset)

        random_with_db_examples = {k: v for d in
                                   [encode_fn(example) for example in tqdm(dataset_requiring_labelling)]
                                   for k, v in d.items()}

        pbar = tqdm(total=dataset_requiring_labelling.num_rows)

        # ThreadPoolExecutor 로 executor가 사용하는 자원들을 적절하게 해제
        # with 문 안에 있는 모든 코드는 동시에 실행되는 스레드 풀의 컨텍스트에서 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            logger.info("Starting LLM processing...")
            # ThreadPoolExecutor를 사용하여 여러 스레드에서 process_LLM 함수를 병렬로 실행
            futures = create_futures(executor=executor,
                                     agent=agent,
                                     propmt_examples=random_with_db_examples,
                                     random_pool_dataset=dataset_requiring_labelling,
                                     db_client=db_client)
            # create_futures 함수에서 생성된 futures를 처리하고, 각 Future의 결과를 수집
            thread_results = process_futures(futures, pbar)  # prompt db

        # with ThreadPoolExecutor 문이 종료되었기 때문에 ThreadPoolExecutor과는 별개로 코드가 실행되며
        # 병렬 처리가 필요하지 않은 코드
        new_label_dataset = []
        for response in tqdm(thread_results, desc="Processing LLM results"):
            ner_result, _ = answer_to_data(response, label_to_id=label_to_id)
            if ner_result:
                new_label_dataset.append(ner_result.raw())
        labelled_dataset_from_api = Dataset.from_list(new_label_dataset)

    if existed_llm_ids:
        existed_llm_data = []
        removed_data_ids = []
        for data_id in existed_llm_ids:
            # 데이터 호출 후 원본 데이터셋 형식에 맞게 반환 ( id, tokens, ner_tags )
            db_answer = db_client.retrieve_memory(f"student:{data_id}:3", "answer")  # prompt db
            if db_answer != "null":
                one_row_data, remove_data_id = answer_to_data(db_answer, label_to_id=label_to_id, data_id=data_id)
                if one_row_data:
                    existed_llm_data.append(one_row_data.raw())
                    removed_data_ids.extend(remove_data_id)
            else:
                # LLM 이 대답하지 못한 경우 DB에는 "null"값이 저장되기 떄문에 이 값을 삭제함
                removed_data_ids.append(data_id)
                logger.info(f"{Fore.RED}{data_id} answer is {db_answer}, So, Remove this from DB{Fore.RESET}")
        if removed_data_ids:
            for id_ in removed_data_ids:
                db_client.delete_key(f"memory:student:{id_}:3")
            logger.info(f"{Fore.YELLOW}Removed {len(removed_data_ids)} data{Fore.RESET}")
        existed_dataset = Dataset.from_list(existed_llm_data)

    # LLM이 태깅할 게 없었다면 DB에 저장된 데이터만 반환 -> DB에 이미 모든 데이터가 저장되어 있을 때
    if len(indices_requiring_labelling) == 0:
        return existed_dataset.cast(sample_pool_dataset.features)
    # LLM이 태깅했고, DB에 저장된 게 없을 때
    elif not existed_llm_ids:
        return labelled_dataset_from_api.cast(sample_pool_dataset.features)
    # 둘 다 있을 때
    else:
        logger.info(f"labelled dataset from api with existed dataset -> {labelled_dataset_from_api.num_rows}")
        result = concatenate_datasets([existed_dataset, labelled_dataset_from_api])
        return result.cast(sample_pool_dataset.features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikiann", choices=["wikiann", "polyglot"])
    parser.add_argument("--dataset_lang", type=str, default="en", choices=["ko", "en", "ja", "pl", "id"])
    parser.add_argument("--model_name", type=str,
                        default="bert-base-multilingual-uncased xlm-roberta-base")  # 기준 모델이 앞에
    # wikiann en : bert 0.8598, xlm 0.8568
    parser.add_argument("--mix_dataset_mode", type=str, default="unlabelled", choices=["original", "unlabelled"])
    parser.add_argument("--portion", type=float, default=1.0)
    parser.add_argument("--ask_oracle", action="store_false")
    parser.add_argument("--inference", action="store_false")
    parser.add_argument("--save_model", action="store_false")

    # python script.py --ask_oracle: 출력은 True
    # python script.py: 출력은 False
    # pycharm 실행 시 store_true -> False

    args = parser.parse_args()

    if args.ask_oracle and args.mix_dataset_mode == "original":
        print("Original dataset does not need LLM. So LLM is not executed.")
        args.ask_oracle = False

    start_time = datetime.datetime.today()

    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=args.portion,
                                      data_mode=args.mix_dataset_mode
                                      )

    model_names_setting_for_log_name = args.model_name.replace('-', '_')
    model_names_setting_for_log_name = model_names_setting_for_log_name.replace(" ", "-")
    logging_file_name = f"models-{model_names_setting_for_log_name}"
    logging_file_name += f"-data_mode_{args.mix_dataset_mode}"
    if args.mix_dataset_mode == "unlabelled":
        logging_file_name += f"-portion_{args.portion}"
    logging_file_name += f"-samples_{data_args.n_samples}"
    logging_file_name += f"-sequence_length_{data_args.max_seq_length}"
    logging_file_name += f"-ask_oracle_{args.ask_oracle}"
    setup_logging(file_name=logging_file_name)
    logging.info(f"Base Arguments: {args}")

    processor = NerProcessor(data_args)
    initial_train_dataset, pool_dataset, eval_dataset, test_dataset = processor.get_dataset()
    logging.info(f"Initial Train dataset size: {initial_train_dataset.num_rows}")

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

    # 데이터 변경 비율 확인
    if args.mix_dataset_mode == "unlabelled" and args.portion == 1.0:
        from itertools import chain

        check_unlabelled = list(chain(*pool_dataset["ner_tags"]))
        assert len(set(check_unlabelled)) == 1, f"Unlabelled dataset has more than one label: {set(check_unlabelled)}"
        logging.info(f"All labels are removed from the dataset.")

    # GPU 메모리 확인
    memory_info = check_gpu_memory()
    gpu_logger = logging.getLogger(f"{check_gpu_memory.__name__}")
    for device, info in memory_info.items():
        gpu_logger.info(f"Device {device}: {info['device_name']}")
        gpu_logger.info(f"Total Memory: {info['total_memory_MiB']} MiB")
        gpu_logger.info(f"Used Memory: {info['used_memory_MiB']} MiB")
        gpu_logger.info(f"Free Memory: {info['free_memory_MiB']} MiB")

        if info["free_memory_MiB"] < 15000:
            gpu_logger.warning(f"{Fore.RED}Device {device} has less than 10,000 MiB of memory left!{Fore.RESET}")
            kill_python_process()

    # model 총집합
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
                                 for model_name in args.model_name.split()})

    db_prompt_client = RedisLLMResponse()
    db_vector_client = RedisVector(dataset_title_value=args.dataset_name,
                                   dataset_lang_value=args.dataset_lang)
    with open("config.yaml") as f:
        agent_config = EasyDict(yaml.safe_load(f))

    peer = Student(config=agent_config, db_client=db_prompt_client,
                   dataset_name=args.dataset_name, dataset_lang=args.dataset_lang)

    current_score = 0
    target_score = 0.9
    tokens_used = 0

    start_threshold = 0.1
    current_step = 0
    total_steps = 30

    while target_score > current_score:
        # 현재 스텝에서의 threshold 계산
        current_threshold = calculate_threshold(start=target_score,
                                                end=start_threshold,
                                                current_step=current_step,
                                                total_steps=total_steps)

        current_step += 1
        if all(trainer_dict["early_stopped"] for trainer_dict in model_trainers.values()):
            logging.warning(f"{Fore.RED}All ModelTrainer instances are early stopped. {Fore.RESET}")
            break

        for model_name, trainer_dict in model_trainers.items():
            trainer = trainer_dict["trainer"]

            embeddings_of_initial_dataset: np.ndarray[np.float32] = \
                trainer.get_embeddings(trainer.copy_of_init_train_dataset)

            # 최초 데이터셋 임베딩 지속적 업데이트
            db_vector_client.insert_vectors(model_name=model_name,
                                            ids=initial_train_dataset["id"],
                                            vectors=embeddings_of_initial_dataset)

            if trainer_dict["early_stopped"]:
                # 이미 학습을 멈춘 모델은 더이상의 학습 진행 없음
                continue

            trainer.train()
            trainer_dict["training_count"] += 1  # 단순 학습 계산

            if trainer.early_stopping_callback.early_stopped:
                # 내부적으로 학습 중단이 판단되었을 경우
                # 학습 중단 플래그가 세워짐
                trainer_dict["early_stopped"] = True
                logging.info(f"{Fore.BLUE}>>>>> {model_name} is early stopped <<<<<{Fore.RESET}")

            eval_result = trainer.evaluate()
            current_score = round(eval_result["eval_f1"], 4)
            # 각 모델의 현재 점수 확인 및 while 루프의 조건 확인
            trainer_dict["current_score"] = current_score

            logging.info(f"{Fore.BLUE}***** Evaluation metrics for {model_name} *****{Fore.RESET}")
            for k, v in eval_result.items():
                logging.info(f"{Fore.BLUE}  {k} = {v}{Fore.RESET}")
            logging.info(
                f"Current F1 for {Fore.CYAN}{model_name}: {eval_result['eval_f1']}{Fore.RESET} // Training count: {trainer_dict['training_count']}")

        # 실험용 데이터: 풀에서 n samples 수 만큼 랜덤하게 데이터 추출: 풀 내부 데이터는 untokenize
        # (index, dataset)을 반환하지만 인덱스는 사용하지 않으므로 버림
        _, sample_pool_dataset = extract_dataset_by_random_sampling(base_dataset=pool_dataset,
                                                                    n_samples=data_args.n_samples)  # 100개

        # 모든 모델이 동의하지 않으면 Uncertainty 라고 판단하며, 각 예측에 대해 모델들 간의 불일치 비율을 계산
        uncertain_indices, disagreement_rate, agreement_dataset = \
            uncertainty_sampling_multi_models(trainers=model_trainers,
                                              dataset=sample_pool_dataset,
                                              id_to_label=id_to_label,
                                              training_threshold=current_threshold)

        logging.info(f"Current Threshold: {current_threshold}")
        logging.info(f"Number of Incomplete Data: {len(uncertain_indices)}")
        logging.info(f"Percentage of disagreement between models for each prediction: {disagreement_rate}")

        if not uncertain_indices:
            # 불확실한 데이터가 없다고 판단한 경우, 추출된 sample 모두는 그대로 다음 학습 대상이 됨
            logging.info(f"No uncertain indices found.")
            new_label_dataset = sample_pool_dataset
        else:
            # 확실한 데이터는 샘플풀에서 제거
            uncertain_sample_pool_dataset = match_indices_from_base_dataset(base_dataset=sample_pool_dataset,
                                                                            indices_to_find=uncertain_indices,
                                                                            remove=False)

            logging.info(f"{sample_pool_dataset.num_rows} samples are extracted from the pool dataset.")
            if args.ask_oracle:
                # 프롬프트에 넣을 비슷한 예제 찾기
                new_label_dataset = get_llm_labeling(agent=peer,
                                                     uncertain_data_indices=uncertain_indices,
                                                     sample_pool_dataset=uncertain_sample_pool_dataset,
                                                     label_to_id=label_to_id,
                                                     db_client=db_prompt_client)

                # 기존 pool dataset 에서 llm 이 보정한 불확실한 데이터셋을 제거
                pool_dataset = match_indices_from_base_dataset(base_dataset=pool_dataset,
                                                               indices_to_find=new_label_dataset["id"] +
                                                                               agreement_dataset["id"])

                # pool dataset 에 새로운 라벨을 업데이트: pool update #1
                pool_dataset = concatenate_datasets([pool_dataset, new_label_dataset, agreement_dataset])
            else:
                # oracle에게 물어볼 데이터셋이 없다면 샘플 데이터셋이 곧 다음 학습 데이터셋
                new_label_dataset = sample_pool_dataset

        for model_name, trainer_dict in model_trainers.items():
            if trainer_dict["early_stopped"]:
                # 모델이 이미 학습을 멈추었다면 다음 학습을 위한 데이터셋 업데이트는 필요없음
                continue
            updated_trainer = trainer_dict["trainer"]
            updated_trainer.update_dataset(new_label_dataset)
            logging.info(
                f"***** Updated training dataset for {model_name} length -> {updated_trainer.train_dataset.num_rows} *****")

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
            for model_name, trainer_dict in model_trainers.items():
                trainer = trainer_dict["trainer"]
                test_result = trainer.get_predictions(test_dataset)

                logging.info(f"***** Test metrics for {model_name} *****")
                for k, v in test_result.metrics.items():
                    logging.info(f"  {k} = {v}")
        else:
            logging.warning(f"Test dataset is not provided.")

    if args.save_model:
        os.makedirs("outputs", exist_ok=True)
        for model_name, trainer_dict in model_trainers.items():
            trainer = trainer_dict["trainer"]
            trainer_dict["trainer"].save_model(os.path.join("outputs", logging_file_name))

        first_tokenizer = model_trainers[list(model_trainers.keys())[0]].trainer.tokenizer
        torch.save(first_tokenizer.train_dataset,
                   os.path.join("outputs", logging_file_name, "train_dataset"))  # 학습 데이터 저장
