import copy
import logging
from functools import partial

import numpy as np

import torch
import datasets
import transformers

import transformers.utils.logging
from datasets import concatenate_datasets
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification
)

from utils.metric import Metrics
from utils.arguments import ModelArguments


@dataclass
class CustomPredictionOutput:
    predictions: np.ndarray
    label_positions: np.ndarray
    input_ids: List[Any]
    metrics: Any
    ids: List[str]
    tokens: Union[str, None] = None
    offset_mapping: Union[np.ndarray, None] = None


def calculate_match_rate(ref_pred, pred, id_to_label):
    # ref_pred.shape == (256, 9)
    assert ref_pred.shape == pred.shape, "The predictions should have the same length"
    ref_entities = np.argmax(ref_pred, axis=-1)
    pred_entities = np.argmax(pred, axis=-1)

    ref_labels = [id_to_label[l] for (p, l) in zip(pred_entities, ref_entities) if l != -100]
    pred_labels = [id_to_label[p] for (p, l) in zip(pred_entities, ref_entities) if l != -100]

    # 모델이 아무것도 "발견"하지 못한 경우로 해석
    # 아무런 엔티티도 찾지 못한 경우에는 항상 match_rate가 하락
    if all(label == 'O' for label in ref_labels) or all(label == 'O' for label in pred_labels):
        return 0

    # DESCRIPTION: ["O", "O", "B-PER"]과 ["O", "O", "B-LOC"] => 2/3
    # DESCRIPTION: ["O", "O", "B-PER"]과 ["O", "B-LOC", "I-LOC"] => 1/3
    match_count = sum(1 for ref, p in zip(ref_labels, pred_labels) if ref == p)
    return match_count / len(ref_labels)  # 일치하는 레이블의 비율을 계산


# start_threshold부터 시작하여 num_bad_epochs가 average_patience에 도달할 때까지 점진적으로 threshold 값을 증가
def calculate_threshold(start: float, end: float, current_step: int, total_steps: int) -> float:
    return start + (end - start) * (current_step / total_steps)


def communicate_models_for_uncertainty(*model_outputs: CustomPredictionOutput,
                                       id_to_label: dict,
                                       threshold: float = 0.5,
                                       ) -> Tuple[List[Any], float, dict]:
    """
    모델의 예측 값들을 비교하여 불확실성을 계산하는 함수8
    disagreement_count:  다른 모델의 예측과 첫 번째 모델의 예측이 다른 경우의 수를 세는 변수
    total_count: 전체 예측 수
    (total_predictions - 1): 비교 대상이 되는 모델의 수 - 첫 번쨰 모델의 예측을 기준으로 다른 모델의 예측을 비교하기에 -1을 함
    """

    assert model_outputs, "No model outputs provided for uncertainty calculation"
    total_predictions = len(model_outputs)
    assert total_predictions > 1, "At least two models are required or put * before the model outputs"
    ref_model = model_outputs[0]
    ref_predictions = model_outputs[0].predictions  # (100, 256, 9) 100개의 문장, 256개의 토큰, 9개의 태그
    ref_ids = model_outputs[0].ids

    disagreement_count = 0
    total_count = len(ref_predictions)  # 예제 문장 수, ref_predictions의 첫 번째 차원의 크기
    disagreement_ids = []
    agreement_id_dict = {}

    for output in model_outputs[1:]:
        # 각 예측 쌍에 대해 일치하지 않는 경우에만 아이디를 추가하는 방식
        # 한 예측 쌍에 대해 불일치가 여러 번 발생해도 아이디는 한 번만 추가되도록 set을 사용
        temp_disagreement_ids = set()
        for idx, (ref_pred, pred, ref_id) in enumerate(zip(ref_predictions, output.predictions, ref_ids)):
            # match_rate 일치하는 비율
            match_rate = calculate_match_rate(ref_pred, pred, id_to_label)
            if match_rate <= threshold:
                disagreement_count += 1
                temp_disagreement_ids.add(ref_id)
            else:
                agreement_id_dict[ref_id] = {"predictions": ref_pred,
                                             "tokens": ref_model.tokens[idx],
                                             "label_positions": ref_model.label_positions[idx],
                                             "offset_mapping": ref_model.offset_mapping[idx]}
        disagreement_ids.extend(temp_disagreement_ids)

    # DESCRIPTION: average_disagreement_rate이 클 수록 모델 간 예측이 자주 일치하지 않음 -> 불확실성이 높음
    # DESCRIPTION: 작을 수록 모델 간 예측이 자주 일치함 -> 불확실성이 낮음
    # DESCRIPTION: 여러 모델의 예측이 얼마나 일관성을 갖는지를 나타내는 지표
    average_disagreement_rate = disagreement_count / total_count / (total_predictions - 1)
    return disagreement_ids, average_disagreement_rate, agreement_id_dict


def get_original_labels(tokens, predictions, label_positions, offset_mapping, id_to_label, tokenizer):
    # 가장 높은 확률을 가진 클래스의 인덱스를 찾습니다.
    predicted_ids = np.argmax(predictions, axis=-1)
    tokenized_tokens = tokenizer.tokenize(" ".join(tokens))

    # 인덱스를 레이블로 변환합니다.
    position_indices = np.where(label_positions != -100)[0].tolist()

    slice_indices = slice(position_indices[0], position_indices[-1] + 1)
    label_positions = label_positions[slice_indices]
    offset_mapping = offset_mapping[slice_indices]
    predicted_ids = predicted_ids[slice_indices]

    predicted_labels = [id_to_label[pred] for label, pred in zip(label_positions, predicted_ids)]

    original_labels = []
    subword_prefix = '▁' if "roberta" in tokenizer.name_or_path else "##"

    for pred, label, offset, token in zip(predicted_labels, label_positions, offset_mapping, tokenized_tokens):
        if offset[0] == 0 and offset[1] != 0:
            if token == subword_prefix and "roberta" in tokenizer.name_or_path:
                continue
            if token.startswith(subword_prefix) and "bert-base" in tokenizer.name_or_path:
                continue
            else:
                original_labels.append(pred)
    return original_labels


# def communicate_models_for_uncertainty(*model_outputs: CustomPredictionOutput,
#                                        id_to_label: dict) -> Tuple[List[Any], float]:
#     """
#     모델의 예측 값들을 비교하여 불확실성을 계산하는 함수
#     disagreement_count:  다른 모델의 예측과 첫 번째 모델의 예측이 다른 경우의 수를 세는 변수
#     total_count: 전체 예측 수
#     (total_predictions - 1): 비교 대상이 되는 모델의 수 - 첫 번쨰 모델의 예측을 기준으로 다른 모델의 예측을 비교하기에 -1을 함
#     """
#     assert model_outputs, "No model outputs provided for uncertainty calculation"
#     total_predictions = len(model_outputs)
#     assert total_predictions > 1, "At least two models are required or put * before the model outputs"
#
#     # 첫 번째 모델 예측 값을 비교를 위한 기준으로 설정
#     ref_predictions = model_outputs[0].predictions
#     ref_ids = model_outputs[0].ids
#
#     disagreement_count = 0
#     total_count = len(ref_predictions)
#     disagreement_ids = []
#
#     # Increment disagreement_count for every prediction that doesn't match with the reference
#     for output in model_outputs[1:]:
#         for ref_pred, pred, ref_id in zip(ref_predictions, output.predictions, ref_ids):
#             ref_entities = np.argmax(ref_pred, axis=-1)
#             pred_entities = np.argmax(pred, axis=-1)
#
#             ref_labels = [id_to_label[l] for (p, l) in zip(pred_entities, ref_entities) if l != -100]
#             pred_labels = [id_to_label[p] for (p, l) in zip(pred_entities, ref_entities) if l != -100]
#
#             if not np.array_equal(ref_labels, pred_labels):
#                 disagreement_count += 1
#                 disagreement_ids.append(ref_id)
#     average_disagreement_rate = disagreement_count / total_count / (total_predictions - 1)
#     return disagreement_ids, average_disagreement_rate


class EarlyStoppingCallbackWithCheck(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=0, early_stopping_threshold=0.0, model_name=None):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.early_stopped = False
        self.prev_f1_score = 0.0
        self.num_bad_epochs = 0
        self.model_name = model_name

        self.logger = logging.getLogger("transformers.early_stopping_callback")
        self.logger.setLevel(logging.DEBUG)

        self.info_logger = logging.getLogger(f"{model_name} Early Stopping Callback")
        self.info_logger.setLevel(logging.DEBUG)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        if metric_to_check.startswith("test_"):
            return
        metric_value = metrics.get(metric_to_check)

        # check if metric_value is None
        if metric_value is None:
            self.logger.warning(
                f"{metric_to_check} does not exist in the evaluation metrics, skipping early stopping check.")
            return

        if metric_value <= self.prev_f1_score + self.early_stopping_threshold:
            self.num_bad_epochs += 1
        else:
            self.num_bad_epochs = 0

        self.info_logger.info(
            f"{self.model_name} {metric_to_check} score: {round(metric_value, 2)}, Number of bad epoch: {self.num_bad_epochs}")

        # 조기 종료 체크
        if self.num_bad_epochs >= self.early_stopping_patience:
            self.early_stopped = True
        else:
            self.early_stopped = False

        self.prev_f1_score = metric_value

    def on_train_end(self, args, state, control, **kwargs):
        if self.early_stopped:
            control.should_training_stop = True


class ModelTrainer(Trainer):
    def __init__(self,
                 model_name_or_path,
                 initial_train_dataset,
                 valid_dataset,
                 data_args,
                 label_list):

        self.logger = logging.getLogger(f"ModelTrainer-{model_name_or_path}")

        self.model_args = ModelArguments(model_name_or_path=model_name_or_path)
        set_seed(self.model_args.seed)

        self.model_type = model_name_or_path.split("-")[0]
        if "/" in self.model_type:
            self.model_type = self.model_type.split("/")[-1]

        self.data_args = data_args
        self.metrics = Metrics(self.data_args, label_list=label_list, do_report=True)

        column_names = initial_train_dataset.column_names

        if "tokens" in column_names:
            self.data_args.text_column_name = "tokens"
        else:
            self.data_args.text_column_name = column_names[0]

        # Use label_column_name if provided, otherwise use the first column with a name not matching "tokens" or
        if self.data_args.label_column_name is not None:
            self.data_args.label_column_name = self.data_args.label_column_name
        if "random_ner_tags" in column_names:
            self.data_args.label_column_name = "random_ner_tags"
        else:
            self.data_args.label_column_name = "ner_tags"

        self.label_to_id = {l: idx for idx, l in enumerate(label_list)}
        self.id_to_label = dict(enumerate(label_list))
        if self.data_args.label_all_tokens:
            self.b_to_i_label = [
                label_list.index(label.replace("B-", "I-")) if label.startswith("B-") else idx
                for idx, label in enumerate(label_list)
            ]

        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 num_labels=len(label_list),
                                                 finetuning_task=data_args.dataset_name,
                                                 label2id=self.label_to_id,
                                                 id2label=self.id_to_label)

        add_prefix_space = True if "roberta" in self.model_type else False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=add_prefix_space)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=self.config)

        self.train_dataset = self.tokenize_dataset(initial_train_dataset)
        self.copy_of_init_train_dataset = copy.deepcopy(self.train_dataset)
        self.eval_dataset = self.tokenize_dataset(valid_dataset, label_column_name="ner_tags")

        self.training_args = TrainingArguments(
            report_to="none",
            learning_rate=self.model_args.learning_rate,
            output_dir="./outputs",
            num_train_epochs=self.model_args.num_train_epochs,
            per_device_train_batch_size=self.model_args.train_batch_size,
            per_device_eval_batch_size=self.model_args.valid_batch_size,
            warmup_steps=self.model_args.warmup_steps,
            weight_decay=self.model_args.weight_decay,
            log_level="info",
            logging_dir=self.model_args.logging_dir,
            save_strategy=self.model_args.save_strategy,
            evaluation_strategy=self.model_args.evaluation_strategy,
            fp16=self.model_args.fp16,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            eval_accumulation_steps=1,
        )

        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        self.early_stopping_callback = EarlyStoppingCallbackWithCheck(
            early_stopping_patience=self.model_args.early_stopping_patience,
            model_name=self.model_args.model_name_or_path
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer,
                                                                padding=True,
                                                                max_length=128,
                                                                pad_to_multiple_of=8 if self.training_args.fp16 else None)
        super().__init__(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[self.early_stopping_callback],
            compute_metrics=self.metrics.compute_metrics,
        )

    def get_embeddings(self, dataset):
        if "input_ids" not in dataset.column_names:
            dataset = self.tokenize_dataset(dataset)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.tensor(dataset["input_ids"]).to(device)
        token_type_ids = torch.tensor(dataset["token_type_ids"]).to(device)
        attention_mask = torch.tensor(dataset["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = self.model.base_model(input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask).last_hidden_state
        return outputs[:, 0, :].cpu().numpy().astype(np.float32)

    def get_predictions(self, dataset):
        self.logger.info("Predictions")
        tokenized_dataset = self.tokenize_dataset(dataset)
        predictions = self.predict(tokenized_dataset)

        custom_prediction_output = CustomPredictionOutput(predictions=predictions.predictions,
                                                          label_positions=predictions.label_ids,
                                                          metrics=predictions.metrics,
                                                          offset_mapping=tokenized_dataset["offset_mapping"],
                                                          input_ids=tokenized_dataset["input_ids"],
                                                          tokens=tokenized_dataset["tokens"],
                                                          ids=dataset["id"])

        return custom_prediction_output

    def train(self):
        self.logger.info(f"{self.model_args.model_name_or_path} is on Training")
        if "input_ids" not in self.train_dataset.column_names:
            self.train_dataset = self.tokenize_dataset(self.train_dataset)
        super().train()

    def update_dataset(self, new_dataset):
        self.logger.info("Update initial train dataset with new dataset")
        tokenized_dataset = self.tokenize_dataset(new_dataset)
        self.train_dataset = concatenate_datasets([self.train_dataset, tokenized_dataset])
        self.train_dataset = self.train_dataset.shuffle(seed=self.model_args.seed)
        self.logger.info(f"Updated train dataset Length is {len(self.train_dataset)}")

    def tokenize_dataset(self, dataset, label_column_name=None, column_names=None):
        label_column_name = label_column_name if label_column_name else self.data_args.label_column_name
        col_names = column_names if column_names else dataset.column_names
        encode_fn = partial(tokenize_and_align_labels,
                            tokenizer=self.tokenizer,
                            label_column_name=label_column_name,
                            model_type=self.model_type,
                            pad_to_max_length=self.data_args.pad_to_max_length,
                            max_seq_length=self.data_args.max_seq_length)
        return dataset.map(encode_fn,
                           load_from_cache_file=False,
                           batched=True
                           )


def tokenize_and_align_labels(examples,
                              label_column_name,
                              tokenizer,
                              model_type,
                              pad_to_max_length="max_length",
                              max_seq_length=128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=pad_to_max_length,
        max_length=max_seq_length,
        return_token_type_ids=True if model_type in ["bert", "xlm"] else False,
        return_offsets_mapping=True,
    )

    all_labels = examples[label_column_name]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id == None:
            # Special token
            new_labels.append(-100)

        elif word_id >= len(labels):
            new_labels.append(-100)

        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = labels[word_id]
            new_labels.append(label)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
