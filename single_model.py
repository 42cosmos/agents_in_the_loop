import datetime
import logging
import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="docent",
                        choices=["wikiann", "polyglot", "mit_restaurant", "mit_movie_trivia", "bionlp2004", "docent"])
    parser.add_argument("--dataset_lang", type=str, default="ko", choices=["ko", "en", "ja", "pl", "id"])
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")  # bert-base-multilingual-uncased
    parser.add_argument("--mix_dataset_mode", type=str, default="original", choices=["original", "unlabelled"])
    parser.add_argument("--portion", type=float, default=1.0)
    args = parser.parse_args()

    # 도메인 데이터셋과 데이터셋 언어 설정이 맞는지 확인
    validate_dataset_language(args)

    start_time = datetime.datetime.now()

    selected_dataset = args.dataset_name

    model_args = ModelArguments(model_name_or_path=args.model_name)
    data_args = DataTrainingArguments(dataset_name=selected_dataset,
                                      dataset_lang=args.dataset_lang,
                                      portion=args.portion,
                                      data_mode=args.mix_dataset_mode
                                      )

    logging_file_name = f"dataset_{args.dataset_name}-{args.dataset_lang}"
    logging_file_name += f"-data_mode_{args.mix_dataset_mode}"
    if args.mix_dataset_mode == "unlabelled":
        logging_file_name += f"-portion_{args.portion}"
    logging_file_name += f"-sequence_length_{data_args.max_seq_length}"
    model_names_setting_for_log_name = args.model_name.replace('-', '_')
    logging_file_name += f"-single_{model_names_setting_for_log_name}"
    setup_logging(file_name=logging_file_name)

    processor = NerProcessor(data_args)

    init_train_dataset, pool_dataset, eval_dataset, test_dataset = processor.get_dataset()
    if "random_ner_tags" in init_train_dataset.column_names:
        init_train_dataset = init_train_dataset.rename_column("random_ner_tags", "ner_tags")

    if "random_ner_tags" in pool_dataset.column_names:
        pool_dataset = pool_dataset.remove_columns("ner_tags")
        pool_dataset = pool_dataset.rename_column("random_ner_tags", "ner_tags")

    init_train_dataset = init_train_dataset.cast(pool_dataset.features)
    train_dataset = concatenate_datasets([init_train_dataset, pool_dataset])

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(label_list))
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)  # BertTokenizerFast

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

    logging.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    for key, value in metrics.items():
        logging.info(f"  {key} = {value}")

    logging.info(f"***** Test *****")
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    for key, value in metrics.items():
        logging.info(f"  {key} = {value}")

    end_time = datetime.datetime.now()
    logging.info(
        f"{Fore.LIGHTMAGENTA_EX}***** {args.model_name} {args.dataset_name}-{LANGUAGES[args.dataset_lang]} Final F1 score: {metrics['test_f1']}*****{Fore.RESET}")
    logging.info(
        f"Starts at {start_time.strftime('%Y-%m-%d %H-%M-%S')} -> Ends at {end_time.strftime('%Y-%m-%d %H-%M-%S')}")
    logging.info(f" 소요 시간: {end_time - start_time}")
