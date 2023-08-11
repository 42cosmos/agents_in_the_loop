from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Early stopping patience"}
    )
    train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for training / evaluation"}
    )
    valid_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for training / evaluation"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for Adam."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    output_dir: Optional[str] = field(
        default="./outputs",
        metadata={"help": "Where do you want to store the your models checkpoint"},
    )
    logging_dir: Optional[str] = field(
        default="logs",
        metadata={"help": "Where do you want to store the your training logs"},
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Log every X updates steps."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The number of epochs to train."},
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Linear warmup over warmup_steps."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use. (no, steps, epochs)"},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint evaluation strategy to use. (no, steps, epochs)"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay if we apply some."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_lang: str = field(
        metadata={"help": "The language of the dataset to use (ex. en, ko ...)."}
    )
    make_data_pool: bool = field(
        default=True,
        metadata={"help": "Whether to make data pool or not."},
    )
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_mode: str = field(default="original", metadata={"help": "The mode of data (train, valid, test)."})
    portion: float = field(default=0.2, metadata={
        "help": "The portion of data to use when you choose random_entity_parital mode. Please input between 0 and 1."})
    data_dir: str = field(default="/home/eunbinpark/workspace/agents_in_the_loop/cached_dataset",
                          metadata={"help": "The directory of data."})
    valid_data_modes = ("original", "random_entity", "random_word_and_entity", "random_entity_partial", "unlabelled")
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: str = field(
        default="max_length",
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    n_samples: int = field(
        default=100,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    initial_train_n_percentage: int = field(
        default=10,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

        if self.data_mode not in self.valid_data_modes:
            raise ValueError(f"Invalid data_mode value: {self.data_mode}. Valid values are {self.valid_data_modes}.")

        if self.data_mode == "random_entity_partial":
            if not (0 <= self.portion <= 1):
                raise ValueError("When data_mode is 'random_entity_partial', 'portion' should be between 0 and 1.")
