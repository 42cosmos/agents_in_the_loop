import datetime
import logging
import argparse
import os
from functools import partial

from colorama import Fore
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    DataCollatorForTokenClassification,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.utils import setup_logging

from utils.trainer import tokenize_and_align_labels
from utils.arguments import ModelArguments, DataTrainingArguments
from utils.metric import Metrics

from utils.data_loader import NerProcessor
from utils.llm.agent import LANGUAGES

from datasets import concatenate_datasets


def validate_dataset_language(args):
    error_msg = "This dataset is only available in {{language}}"
    if args.dataset_name == "docent":
        if args.dataset_lang != "ko":
            raise ValueError(error_msg.format(language=LANGUAGES[args.dataset_lang]))
    elif args.dataset_name in ["mit_restaurant", "mit_movie_trivia", "bionlp2004"]:
        if args.dataset_lang != "en":
            raise ValueError(error_msg.format(language=LANGUAGES[args.dataset_lang]))

    if args.ask_oracle and args.mix_dataset_mode == "original":
        raise ValueError("Original dataset does not need LLM. So LLM is not executed.")


def find_logging_file_name(args):
    model_names_for_log_name = args["model_names"].replace('-', '_')
    model_names_for_log_name = model_names_for_log_name.replace(" ", "-")
    logging_file_name = f"models-{model_names_for_log_name}"
    logging_file_name += f"-dataset_{args['dataset_name']}_{args['dataset_lang']}"
    logging_file_name += f"-data_mode_{args['mix_dataset_mode']}"
    if args['mix_dataset_mode'] == "unlabelled":
        logging_file_name += f"-portion_{args['portion']}"
    logging_file_name += f"-samples_{args['n_samples']}"
    logging_file_name += f"-sequence_length_{args['max_seq_length']}"
    logging_file_name += f"-ask_oracle_{args['ask_oracle']}"
    return logging_file_name


def find_cached_train_dataset_path(args, standard_model_name_alias):
    import os
    dir_name = find_logging_file_name(args)
    train_dataset_output_dir = args["output_dir"]
    train_dataset_output_path = os.path.join(train_dataset_output_dir,
                                             dir_name,
                                             standard_model_name_alias,
                                             "train_dataset")
    if os.path.isfile(train_dataset_output_path):
        return train_dataset_output_path
    else:
        raise ValueError(f"{train_dataset_output_path} is not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mit_restaurant",
                        choices=["wikiann", "polyglot", "mit_restaurant", "mit_movie_trivia", "bionlp2004", "docent"])
    parser.add_argument("--dataset_lang", type=str, default="en", choices=["ko", "en", "ja", "pl", "id"])
    parser.add_argument("--model_names", type=str, default="bert-base-multilingual-uncased xlm-roberta-base")
    parser.add_argument("--standard_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--mix_dataset_mode", type=str, default="unlabelled", choices=["original", "unlabelled"])
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--portion", type=float, default=1.0)
    parser.add_argument("--ask_oracle", action="store_false")
    parser.add_argument("--use_train_dataset_in_arch", action="store_false")
    parser.add_argument("--use_original_num_in_arch", action="store_true")

    args = parser.parse_args()
    validate_dataset_language(args)  # 도메인 데이터셋과 데이터셋 언어 설정이 맞는지 확인

    # 파일 실행 시 store_false는 명시했을 때 True 값을 지님
    # pycharm 실행 시 store_true -> False
    # pycharm 실행 시 store_false -> True

    start_time = datetime.datetime.now()

    model_args = ModelArguments(model_name_or_path=args.standard_model)
    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=args.portion,
                                      data_mode=args.mix_dataset_mode,
                                      make_data_pool=True
                                      )

    logging_file_name = f"dataset-{args.dataset_name}_{args.dataset_lang}"
    logging_file_name += f"&data_mode-{args.mix_dataset_mode}"
    if args.mix_dataset_mode == "unlabelled":
        logging_file_name += f"&portion-{args.portion}"
    logging_file_name += f"&sequence_length-{data_args.max_seq_length}"

    train_dataset_path = None
    use_original_num_in_arch = False
    logging_dir = ""

    if args.use_train_dataset_in_arch and args.use_original_num_in_arch:
        raise "use_train_dataset_in_arch and use_original_num_in_arch cannot be True at the same time"

    elif (arch_result_dataset := args.use_train_dataset_in_arch and not args.use_original_num_in_arch) \
            or (arch_results_original_dataset := not args.use_train_dataset_in_arch and args.use_original_num_in_arch):
        all_args = vars(args) | vars(model_args) | vars(data_args)
        train_dataset_path = find_cached_train_dataset_path(all_args,
                                                            standard_model_name_alias=args.standard_model.split("-")[0])

        if arch_result_dataset:
            logging.info(
                f"{Fore.GREEN} {args.dataset_name}-{args.dataset_lang} 아키텍쳐 내부에서 사용된 데이터셋으로 학습을 진행합니다.{Fore.RESET}")
            logging_file_name += "&use_train_dataset_in_arch"
            logging_dir = "train_arch_result"

        elif arch_results_original_dataset:
            logging.info(
                f"{Fore.GREEN} {args.dataset_name}-{args.dataset_lang} 아키텍쳐 내부에서 사용된 데이터셋의 원본으로 학습을 진행합니다.{Fore.RESET}")
            logging_file_name += "&use_original_num_in_arch"
            use_original_num_in_arch = True
            logging_dir = "train_origin_in_arch"

    else:
        logging.info(
            f"{Fore.GREEN} {args.dataset_name}-{args.dataset_lang}: {args.mix_dataset_mode}의 데이터셋로 학습을 진행합니다.{Fore.RESET}")
        logging_file_name += "&single_dataset"
        logging_dir = "train_unlabelled"

    model_names_setting_for_log_name = args.standard_model.replace('-', '_')
    logging_file_name += f"-{model_names_setting_for_log_name}"

    setup_logging(
        logging_path=os.path.join("/home/eunbinpark/workspace/_agent_in_the_loop/final_logs/", logging_dir),
        file_name=logging_file_name)

    if args.split == 20000:
        processor = NerProcessor(data_args, split=args.split)
    else:
        processor = NerProcessor(data_args)

    all_dataset = processor.get_dataset(train_dataset_path=train_dataset_path,
                                        use_original_num_in_arch=use_original_num_in_arch)
    if not args.use_train_dataset_in_arch and not args.use_original_num_in_arch:
        train_dataset, pool_dataset, eval_dataset, test_dataset = all_dataset
        train_dataset = concatenate_datasets([train_dataset, pool_dataset])
        if args.mix_dataset_mode == "unlabelled":
            from itertools import chain

            if not set(list(chain(*[i["ner_tags"] for i in pool_dataset]))) == {0}:
                raise ValueError("Pool dataset must be unlabelled")
    else:
        train_dataset, eval_dataset, test_dataset = all_dataset

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(label_list))
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    col_names = train_dataset.column_names
    encode_fn = partial(tokenize_and_align_labels,
                        tokenizer=tokenizer,
                        label_column_name="ner_tags",
                        model_type=model_args.model_name_or_path.split("-")[0],
                        pad_to_max_length=data_args.pad_to_max_length,
                        max_seq_length=data_args.max_seq_length)

    train_dataset = train_dataset.map(encode_fn, batched=True, remove_columns=col_names)
    eval_dataset = eval_dataset.map(encode_fn, batched=True, remove_columns=col_names)
    test_dataset = test_dataset.map(encode_fn, batched=True, remove_columns=col_names)

    training_args = TrainingArguments(
        output_dir=model_args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=True,
        save_strategy="steps",
        evaluation_strategy="steps",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer,
                                                       pad_to_multiple_of=8 if training_args.fp16 else None)

    metrics = Metrics(data_args, label_list=label_list)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
    )

    train_results = trainer.train()
    logging.info(f"Training results: {train_results}")
    # trainer.save_model() -> Saves the tokenizer too for easy upload

    logging.info("***** Evaluate *****")
    metrics = trainer.evaluate()

    for key, value in metrics.items():
        logging.info(f"  {key} = {value}")

    logging.info(f"***** Test *****")
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    for key, value in metrics.items():
        logging.info(f"  {key} = {value}")

    end_time = datetime.datetime.now()
    if args.use_train_dataset_in_arch:
        logging.info(
            f"{Fore.LIGHTGREEN_EX} train_dataset: {train_dataset.num_rows:,} =>\t"
            f"***** {args.standard_model} {args.dataset_name}-{LANGUAGES[args.dataset_lang]} "
            f"Final F1 score: {metrics['test_f1']}*****{Fore.RESET}")
    else:
        logging.info(
            f"{Fore.LIGHTMAGENTA_EX}***** {args.standard_model} {args.dataset_name}-{LANGUAGES[args.dataset_lang]} "
            f"Final F1 score: {metrics['test_f1']}*****{Fore.RESET}")
    logging.info(
        f"Starts at {start_time.strftime('%Y-%m-%d %H-%M-%S')} -> Ends at {end_time.strftime('%Y-%m-%d %H-%M-%S')}")
    logging.info(f" 소요 시간: {end_time - start_time}")
