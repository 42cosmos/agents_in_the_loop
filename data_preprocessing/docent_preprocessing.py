import glob
import re
import pdb
from tqdm import tqdm

import pandas as pd
from kss import split_sentences


# def remove_ner_tags(text):
#     return re.sub(r'<([^>]+):[^>]*>', r'\1', text)


def swap_tag_numbers(df):
    df["ner_tags"].replace(12, 15, inplace=True)
    df["ner_tags"].replace(13, 12, inplace=True)
    df["ner_tags"].replace(14, 13, inplace=True)
    df["ner_tags"].replace(15, 14, inplace=True)
    return df


def add_id(example, idx, prefix="docent:ko"):
    if prefix is not None:
        idx = f"{prefix}:{idx}"

    example['id'] = idx
    return example


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

entire_df = pd.DataFrame(columns=["tokens", "ner_tags"])

for type_ in ["train", "validation", "test"]:
    docent_list = glob.glob(f"docent/{type_}/*.json")

    for file in tqdm(docent_list, desc=f"{type_.title()} processing files"):
        df = pd.read_json(file)

        # swap tag numbers
        df = swap_tag_numbers(df)

        # full sentence
        full_paragraph = "".join(df["tokens"])

        # split a full paragraph into individual sentences
        split_sentence = split_sentences(full_paragraph)

        df["ner_tags"] = df["ner_tags"].apply(lambda x: docent_tag[int(x)])

        start_idx = 0
        end_idx = 0
        for sentence_idx in range(len(split_sentence)):
            sentence = split_sentence[sentence_idx]
            if sentence_idx != (len(split_sentence)-1):
                sentence += " "

            end_idx += len(sentence)
            tokens = df["tokens"].iloc[start_idx:end_idx].tolist()
            ner_tags = df["ner_tags"].iloc[start_idx:end_idx].tolist()
            start_idx = end_idx

            if sentence_idx != (len(split_sentence)-1):
                tokens = tokens[:-1]
                ner_tags = ner_tags[:-1]
            entire_df.loc[len(entire_df.index)] = [tokens, ner_tags]

entire_df.to_csv("./entire_data.csv", index=False)

print(f"Entire NER tags: {entire_df.shape[0]}")
