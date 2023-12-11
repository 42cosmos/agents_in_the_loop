import yaml
import logging
import argparse

from colorama import Fore
from datasets import concatenate_datasets, Dataset
from easydict import EasyDict

from utils.data_loader import NerProcessor
from utils.arguments import DataTrainingArguments
from utils.llm.base import MessageFunctionCall, Message
from utils.database.redis_client import RedisVector, RedisLLMResponse
from utils.llm.agent import Student, get_example
from utils.llm.token_counter import count_message_tokens, count_string_tokens

import warnings
from utils.llm.agent import LANGUAGES
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.CRITICAL)


def make_example(tokens, ner_tags, id_to_label=None):
    if id_to_label is not None:
        response = {"response": [{"word": token, "entity": id_to_label[tag]} for token, tag in zip(tokens, ner_tags)]}
    response = {"response": [{"word": token, "entity": tag} for token, tag in zip(tokens, ner_tags)]}
    return MessageFunctionCall(role="function",
                               name="find_ner",
                               content=response)


def get_token_usage(example):
    data_id = example["id"]
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    trigger_prompt = str(tokens)
    prompt = Message(role="user", content=trigger_prompt)
    prompt_tokens = count_message_tokens([prompt])

    input_example: dict = make_example(tokens, ner_tags, id_to_label=id_to_label)
    completion_tokens = count_string_tokens(str(input_example.content), "gpt-3.5-turbo-0613")

    return {"id": data_id,
            "input_example": input_example.content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="polyglot",
                        choices=["wikiann", "polyglot", "mit_restaurant", "mit_movie_trivia", "bionlp2004"])

    parser.add_argument("--dataset_lang", type=str, default="en",
                        choices=["ko", "en", "ja", "pl", "id"])

    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-uncased")

    parser.add_argument("--mix_dataset_mode", type=str, default="original",
                        choices=["original", "random_entity", "random_word_and_entity", "random_entity_partial",
                                 "unlabelled"])
    parser.add_argument("--portion", type=float, default=1.0)
    args = parser.parse_known_args()[0]
    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=1,
                                      data_mode=args.mix_dataset_mode
                                      )

    processor = NerProcessor(data_args, split=20000)
    initial_train_dataset, train_dataset, eval_dataset, test_dataset = processor.get_dataset()
    logging.info(f"Initial Train dataset size: {initial_train_dataset.num_rows}")

    label_to_id, id_to_label = processor.label_to_id, processor.id_to_label
    label_list = processor.labels

    db_client = RedisVector(dataset_title_value=args.dataset_name, dataset_lang_value=args.dataset_lang)
    prompt_client = RedisLLMResponse()
    with open("config.yaml") as f:
        agent_config = EasyDict(yaml.safe_load(f))
    peer = Student(config=agent_config, db_client=prompt_client,
                   dataset_name=args.dataset_name, dataset_lang=args.dataset_lang)

    # 2개 예제 ( 고정 )
    basic_example = train_dataset.select(range(2))
    similar_examples = get_example(basic_example, id_to_label)
    two_shots = similar_examples * 2

    two_shots_tokens = sum(count_string_tokens(str(shot.content), "gpt-3.5-turbo-0613") for shot in two_shots)

    # system role 설명, ner desc * 3
    student_prompt = peer._get_prompt(triggering_prompt="test")
    teacher_prompt = peer.teacher._get_prompt(triggering_prompt="test")[:-1]

    system_role_with_ner_guide = student_prompt[:-1] * 2
    system_role_with_ner_guide += teacher_prompt
    system_role_with_ner_guide_tokens = count_message_tokens(system_role_with_ner_guide)

    teacher_keys = db_client.get_all_key_by_pattern("memory*1")
    teacher_keys = list(map(lambda x: ":".join(x.split(":")[1:]), teacher_keys))

    feedbacks = [prompt_client.retrieve_memory(key) for key in teacher_keys]
    token_counts = [count_string_tokens(feedback, "gpt-3.5-turbo-0613") for feedback in feedbacks]

    # peer2 답변
    feedback_tokens = sum(token_counts) / len(token_counts)

    token_usages = train_dataset.map(get_token_usage)

    train_all_input_tokens = sum(token_usages["prompt_tokens"])
    train_all_output_tokens = sum(token_usages["completion_tokens"])

    input_values = [system_role_with_ner_guide_tokens, two_shots_tokens, feedback_tokens]
    input_values_summary = int(sum(input_values) * train_dataset.num_rows)
    input_values_summary += train_all_input_tokens * 3
    input_values_summary += train_all_output_tokens * 2

    output_values = [feedback_tokens]
    output_values_summary = int(sum(output_values) * train_dataset.num_rows)
    output_values_summary += train_all_output_tokens * 2

    print(f"Approximate tokens and cost of {Fore.YELLOW}{args.dataset_name} {LANGUAGES[args.dataset_lang]} Dataset{Fore.RESET}: "
          f"{train_dataset.num_rows:,} examples")
    print(" ".join(token_usages[0]["tokens"]))
    print("=" * 50)
    print(f"Input values summary: {input_values_summary:,}")
    print(f"Output values summary: {output_values_summary:,}")
    print("-" * 30)
    # input_values_summary * 0.0000015, output_values_summary * 0.000002
    print(f"Input cost: {input_values_summary * 0.0000015:,}")
    print(f"Output cost: {output_values_summary * 0.000002:,}")
    print("=" * 50)
    total_cost = input_values_summary * 0.0000015 + output_values_summary * 0.000002
    print(f"Total cost: {round(total_cost, 2):,}")
    total_30 = total_cost * 0.3
    print(f"30% of Total cost: {round(total_30, 2):,}")
