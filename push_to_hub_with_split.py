from datasets import Value, Sequence, ClassLabel, Features, DatasetDict, Dataset, load_dataset
from itertools import chain
from collections import defaultdict
from tqdm import tqdm


def add_id(example, idx):
    example['id'] = idx
    return example


def push_to_my_hub(dataset_name, language, hub_name: str, test_size=0.3):
    cloud_dataset = load_dataset(dataset_name, language)
    assert len(cloud_dataset.keys()) == 1, "You don't need to split the dataset"

    cloud_dataset = cloud_dataset if "id" in cloud_dataset["train"].features \
        else cloud_dataset.map(add_id, with_indices=True)

    ner_list = set(list(chain(*cloud_dataset["train"]["ner"])))
    ner_list = ner_list - {"O"}
    label_list = ["O"] + sorted(list(ner_list))

    dataset_features = Features({
        "id": Value(dtype="string"),
        "tokens": Value(dtype="string"),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })

    # TODO: polyglot 외에도 다른 데이터에 맞게끔 변경
    new_dataset_dict = defaultdict(lambda: dict)
    for i in tqdm(cloud_dataset["train"]):
        new_dataset_dict[i["id"]] = {"id": i["id"],
                                     "tokens": i["words"],
                                     "ner_tags": i["ner"]}

    new_dataset = Dataset.from_list(list(new_dataset_dict.values()), features=dataset_features)

    train_test_valid = new_dataset.train_test_split(test_size=test_size, shuffle=True)
    test_valid = train_test_valid["test"].train_test_split(test_size=0.5, shuffle=True)

    dataset_to_push = DatasetDict({
        "train": train_test_valid["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    })

    dataset_to_push.push_to_hub(hub_name)
