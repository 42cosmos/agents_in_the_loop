import argparse
import os

import torch
from colorama import Fore

from utils.arguments import DataTrainingArguments, ModelArguments
from utils.database.redis_client import RedisLLMResponse
from utils.llm.agent import LANGUAGES
from utils.main_utils import match_indices_from_base_dataset


def load_cached_dataset(data_path, return_model_information=False):
    result = torch.load(data_path)
    if return_model_information:
        model_info = data_path.split("/")[-1].split("-")[0]
        return model_info, result
    return result


def load_initial_dataset(path, dataset, language):
    filename = f"cached-{dataset}-{language}_unlabelled-seq_len_128-10%"
    if dataset.startswith("polyglot"):
        filename = f"TEST_20000::{filename}"
    result = torch.load(os.path.join(path, filename))
    return result[0]


def summarise_token_usages(key, db_client):
    prompt_tokens_usage = db_client.retrieve_memory(key, "prompt_tokens_usage")
    completion_tokens_usage = db_client.retrieve_memory(key, "completion_tokens_usage")
    api_cost = int(prompt_tokens_usage) * 0.0000015 + int(completion_tokens_usage) * 0.000002
    return int(prompt_tokens_usage), int(completion_tokens_usage), api_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikiann",
                        choices=["mit_restaurant", "mit_movie_trivia", "bionlp2004", "wikiann", "polyglot", "docent"])
    parser.add_argument("--dataset_lang", type=str, default="en", choices=["ko", "en", "ja", "pl", "id"])
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-uncased xlm-roberta-base")
    parser.add_argument("--output_path", type=str, default="/home/eunbinpark/workspace/agents_in_the_loop/outputs")

    args = parser.parse_args()

    data_args = DataTrainingArguments(dataset_name=args.dataset_name,
                                      dataset_lang=args.dataset_lang,
                                      portion=1.0,
                                      data_mode="unlabelled")

    model_output_path = args.output_path if args.output_path else ModelArguments.output_dir

    model_names_setting_for_log_name = args.model_name.replace('-', '_')
    model_names_setting_for_log_name = model_names_setting_for_log_name.replace(" ", "-")
    dir_name = f"models-{model_names_setting_for_log_name}"
    dir_name += f"-dataset_{args.dataset_name}_{args.dataset_lang}"
    dir_name += f"-data_mode_unlabelled"
    dir_name += f"-portion_1.0"
    dir_name += f"-samples_{data_args.n_samples}"
    dir_name += f"-sequence_length_{data_args.max_seq_length}"
    dir_name += f"-ask_oracle_True"

    print(dir_name)
    redis = RedisLLMResponse()
    initial_train_dataset = load_initial_dataset(data_args.data_dir, args.dataset_name, args.dataset_lang)
    dataset_names = [os.path.join(model.split("-")[0], f"train_dataset") for model in args.model_name.split(" ")]
    output_data_pathes = [os.path.join(model_output_path, dir_name, model) for model in dataset_names]
    used_datasets_by_name = [load_cached_dataset(output_data_path, return_model_information=True) for output_data_path
                             in output_data_pathes]

    print(f"{Fore.GREEN}***** {args.dataset_name} with {LANGUAGES[args.dataset_lang]} ***** ")
    print(f"Initial train dataset: {initial_train_dataset.num_rows}{Fore.RESET}")
    for idx, used_dataset in enumerate(zip(args.model_name.split(" "), used_datasets_by_name)):
        model_full_name = used_dataset[0]
        model_short_name, dataset = used_dataset[1]
        extracted_from_initial_dataset = match_indices_from_base_dataset(dataset,
                                                                         initial_train_dataset["id"])

        colour = Fore.YELLOW if idx % 2 == 0 else Fore.CYAN
        print(
            f"{colour}{model_full_name} -> {extracted_from_initial_dataset.num_rows} datum used in train!")

        in_db = []
        find_key = f"memory*{{dataset_id}}:3"
        for dataset_id in extracted_from_initial_dataset["id"]:
            find_db_key = find_key.format(dataset_id=dataset_id)
            in_db.extend(redis.get_all_key_by_pattern(find_db_key))

        print(f"{model_full_name} -> {len(in_db)} LLM Responses in Database ! {Fore.RESET}")
        prompt_token_usage, completion_token_usage, api_cost = zip(
            *map(lambda x: summarise_token_usages(x, redis), in_db))

        print(f"*** ${round(sum(api_cost), 2)} ***\n"
              f"*** Prompt Token Usage -> {sum(prompt_token_usage):,}\n"
              f"*** Completion Token Usage {sum(completion_token_usage):,}")
