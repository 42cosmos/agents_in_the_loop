import pdb
import ast

from datasets import Value, Sequence, ClassLabel, Features, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

import pandas as pd


def add_id(example, idx, prefix="docent:ko"):
    if prefix is not None:
        idx = f"{prefix}:{idx}"

    example['id'] = idx
    return example

entire_df = pd.read_csv("./entire_data.csv", sep=',', quotechar='"')

# String to List
entire_df['ner_tags'] = entire_df['ner_tags'].apply(lambda x: eval(x))
entire_df['tokens'] = entire_df['tokens'].apply(lambda x: ast.literal_eval(x))


# docent tag list
docent_tag = [
    'B-DT',
    'I-DT',
    'B-LC',
    'I-LC',
    'B-OG',
    'I-OG',
    'B-PS',
    'I-PS',
    'B-QT',
    'I-QT',
    'B-TI',
    'I-TI',
    'B-DUR',
    'I-DUR',
    'O',
]

# split into train, valid, test
train_df, temp_df = train_test_split(entire_df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# reset index
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


# set dataset features
dataset_features = Features({
        "tokens": Sequence(Value(dtype="string")),
        "ner_tags": Sequence(ClassLabel(names=docent_tag))
    })

train_dataset = Dataset.from_pandas(train_df, features=dataset_features)
valid_dataset = Dataset.from_pandas(valid_df, features=dataset_features)
test_dataset = Dataset.from_pandas(test_df, features=dataset_features)

train_dataset = train_dataset.map(add_id, with_indices=True)
valid_dataset = valid_dataset.map(add_id, with_indices=True)
test_dataset = test_dataset.map(add_id, with_indices=True)


dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
})

dataset_dict = dataset_dict.map(add_id, with_indices=True)

# Push to hub
dataset_dict.save_to_disk("./local_dataset")
dataset_dict.push_to_hub("eunbincosmos/docent-ko")