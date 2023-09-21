import json
import re
import glob
from tqdm import tqdm

import kss

from datasets import Value, Sequence, ClassLabel, Features, DatasetDict, Dataset, load_dataset


def add_id(example, idx, prefix="docent:ko"):
    if prefix is not None:
        idx = f"{prefix}:{idx}"

    example['id'] = idx
    return example


def renew_tags(ner_tags: list):
    new_tags = []
    for tag in ner_tags:
        if tag == 12:
            tag = 15
        if tag == 13:
            tag = 12
        if tag == 14:
            tag = 13
        new_tags.append(tag)
    results = []
    for tag in new_tags:
        if tag == 15:
            tag = 14

        results.append(tag)
    return results


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def show_test(test, num=None):
    if not num:
        for idx, t in enumerate(test):
            tokens = t["tokens"]
            ner_tags = t["ner_tags"]
            print(f"{'+' * 15}{idx}{'+' * 15}")
            for token, tag in zip(tokens, ner_tags):
                print(f"{token:1} -> {id_to_label[tag]}")
            print("-" * 30)
    else:
        data = test[num]
        tokens = data["tokens"]
        ner_tags = data["ner_tags"]
        print(f"{'+' * 15}{num}{'+' * 15}")
        for token, tag in zip(tokens, ner_tags):
            print(f"{token:1} -> {id_to_label[tag]}")


def find_spacebar(prev_sent, original_text):
    if not prev_sent:
        return False
    # need_to_find_text의 마지막 부분을 찾습니다.
    end_index = original_text.find(prev_sent) + len(prev_sent)

    # 바로 다음 문자를 찾습니다.
    next_char = original_text[end_index:end_index + 1]

    # 다음 문자가 스페이스인지 아닌지 확인합니다.
    if next_char == ' ':
        return True
    else:
        return False


if __name__ == "__main__":
    docent_files = glob.glob("/home/eunbinpark/workspace/docent/*/*.json")
    docents = [read_json(path) for path in docent_files]

    label_list = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG',
                  'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'B-DUR', 'I-DUR', 'O']
    id_to_label = dict(enumerate(label_list))

    failed_data = []
    final_data = []
    for number, data in enumerate(tqdm(docents)):
        tokens = data["tokens"]
        ner_tags = data["ner_tags"]
        assert len(tokens) == len(ner_tags), "데이터가 잘못됨"
        joined_text = "".join(tokens)

        split_results = kss.split_sentences(joined_text)
        new_tags = renew_tags(ner_tags)

        new_data = []
        for idx, sent in enumerate(split_results):
            temp_text = re.sub(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]', 'X', joined_text)
            temp_search_text = re.sub(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]', 'X', sent)

            start_index = temp_text.index(temp_search_text)
            end_index = start_index + len(sent)

            does_same_length = joined_text[start_index:end_index] == sent

            if not does_same_length:
                break

            split_tags = new_tags[start_index:end_index]

            if len(split_tags) != len(list(sent)):
                failed_data.append(number)
                print(number)
                new_data = []
            else:
                new_data.append({"id": number, "tokens": list(sent), "ner_tags": split_tags})

        final_data.extend(new_data)

        dataset_features = Features({
            "id": Value(dtype="string"),
            "tokens": Sequence(feature=Value(dtype="string")),
            "ner_tags": Sequence(ClassLabel(names=label_list))
        })

        docent_all = Dataset.from_list(final_data, features=dataset_features)
        train_test_valid = docent_all.train_test_split(test_size=0.3, shuffle=True)
        test_valid = train_test_valid["test"].train_test_split(test_size=0.5, shuffle=True)

        dataset_to_push = DatasetDict({
            "train": train_test_valid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        })

        push_to_hub_dataset = dataset_to_push.map(add_id, with_indices=True)
        push_to_hub_dataset.push_to_hub("cosmos42/docent-ko")
