from __future__ import annotations

import os
import json
import logging
import logging.config
from typing import List

from colorama import Fore
from datasets import Dataset
from dotenv import load_dotenv
from dataclasses import dataclass, field

import openai

from .chat import chat_with_agent
from .base import (
    Message,
    MessageRole,
    OpenAIFunctionCall,
    MessageFunctionCall,
    EntityAgentResponse, ChatSequence
)
from .openai import OPEN_AI_CHAT_MODELS

from .token_counter import count_message_tokens
from ..throttling import TokenThrottling

logger = logging.getLogger(f"{__name__}")

LANGUAGES = {
    "en": "english",
    "ko": "korean",
    "ja": "japanese",
    "pl": "polish",
    "id": "indonesian",
}


@dataclass
class MessageHistory:
    agent: Agent
    messages: list[Message] = field(default_factory=list)

    def __getitem__(self, i: int):
        return self.messages[i]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def add(
            self,
            role: MessageRole,
            content: str,
            function_call: OpenAIFunctionCall | None = None,
    ):
        return self.append(Message(role, content, function_call))

    def append(self, message: Message):
        return self.messages.append(message)


class Agent:
    def __init__(self, role,
                 config,
                 db_client,
                 dataset_name: str,
                 dataset_lang: str):

        self.role = role
        self.raw_config = config

        self.config = config[role]
        self.db_client = db_client
        self.dataset_name = dataset_name
        self.dataset_lang = dataset_lang

        self.ner_guidance = self._get_basic_prompt()

        self.logger = logging.getLogger(f"{self.role}")
        self.logger.setLevel(logging.INFO)

        assert dataset_lang in LANGUAGES, f"dataset_lang should be one of {LANGUAGES.keys()}"

        self.api_cost_accumulated = 0.0

        load_dotenv(dotenv_path=self.config.env_path) if self.config.env_path is not None else load_dotenv()
        self.max_conversation_limit = self.config.max_conversation_limit
        self.history = MessageHistory(self)

        system_prompt_key_for_redis = f"prompt:{self.role}:base_system"
        self.system_prompt: str = self.db_client.get_value(system_prompt_key_for_redis)

        self.logger = logging.getLogger("Agent")
        if self.system_prompt is None:
            raise ValueError(f"Please set system_prompt for {self.role} agent")

        self.logger.info(f"{Fore.YELLOW}{role.upper()} Agent{Fore.RESET} is Ready to talk !")

        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "Please set OPENAI_API_KEY in .env file"
        openai.api_key = self.api_key

        self.organisation_key = os.getenv("ORGANISATION_KEY")
        assert self.organisation_key is not None, "Please set ORGANISATION_KEY in .env file"
        if self.organisation_key:
            openai.organization = self.organisation_key

    def _get_basic_prompt(self):
        information = self.db_client.get_value(f"prompt:{self.dataset_name}:{LANGUAGES[self.dataset_lang]}")
        warning_sign = ""
        if self.role == "student":
            warning_sign = self.db_client.get_value(f"prompt:warning")

        elif self.role == "teacher":
            warning_sign = self.db_client.get_value(f"prompt:teacher:warning")
        return f"{information}\n{warning_sign}"

    def _get_test_basic_prompt(self, triggering_prompt, examples: List[MessageFunctionCall] = None):
        """
        Message 예시를 보여주는 코드, main 에서는 사용되지 않는다
        :param triggering_prompt:
        :return:
        """

        message_sequence = ChatSequence.for_model(
            self.config.model_name,
            [
                Message("system", self.system_prompt),
                Message("user", self.ner_guidance),
                Message("user", triggering_prompt)
            ],
        )

        if examples is None:
            return message_sequence

        message_sequence.append(examples)
        return message_sequence

    def create_chat_with_agent(self,
                               raw_question_data,
                               user_sentence,
                               function_call_examples: List[MessageFunctionCall] = None,
                               throttling: TokenThrottling = None,
                               expected_return_tokens: int = 0):
        assert user_sentence, "Please set your question"

        assistant_reply = chat_with_agent(
            agent=self,
            raw_question_data=raw_question_data,
            system_prompt=self.system_prompt,
            llm_guidance=self.ner_guidance,
            triggering_prompt=user_sentence,
            function_call_examples=function_call_examples,
            expected_return_tokens=expected_return_tokens,
            throttling=throttling
        )
        return assistant_reply


class Teacher(Agent):
    def __init__(self, config, db_client, dataset_name, dataset_lang):
        super().__init__("teacher", config, db_client, dataset_name, dataset_lang)
        self.logger = logging.getLogger(__class__.__qualname__)

    def give_feedback(self, raw_data, student_reply, throttling: TokenThrottling = None):
        if student_reply is None or student_reply.tokens is None:
            self.logger.info(f"{raw_data['id']} has no answer")
            prompt = f"""There is no answer given by the student.\n Question Sentence: {raw_data["tokens"]}"""

        else:
            problem_sentence = student_reply.tokens
            student_answer = student_reply.ner_tags
            prompt = f"""Here's the sentence in which the student has to find the entity, and the answer given by the student:\nQuestion Sentence: {str(problem_sentence)}\nStudent's Answer: "{student_answer}" """.strip()
        assistant_reply = self.create_chat_with_agent(raw_question_data=raw_data,
                                                      user_sentence=prompt,
                                                      throttling=throttling)

        return assistant_reply


class Student(Agent):
    def __init__(self, config, db_client, dataset_name, dataset_lang):
        super().__init__("student", config, db_client, dataset_name, dataset_lang)
        self.teacher = Teacher(config=self.raw_config,
                               db_client=db_client,
                               dataset_name=dataset_name,
                               dataset_lang=dataset_lang)

        self.logger = logging.getLogger(__class__.__qualname__)
        model_info = OPEN_AI_CHAT_MODELS[self.config.model_name]
        # https://platform.openai.com/account/rate-limits
        request_per_second = model_info.RPM / 60
        token_per_second = model_info.TPM / 60
        self.throttling = TokenThrottling(rate=request_per_second, token_rate=token_per_second)

    def talk_to_agent(self,
                      raw_data,
                      db_prompt_client,
                      similar_examples=None,
                      get_feedback=False,
                      save_db=True):
        assert (save_db and db_prompt_client is not None), "Please set db_client if you want to save the conversation"
        get_predicted_token_usage = count_expected_tokens(raw_data["tokens"])

        self.logger.info(f"Processing example: {raw_data['id']}")
        answers = [self.create_chat_with_agent(user_sentence=str(raw_data["tokens"]),
                                               raw_question_data=raw_data,
                                               function_call_examples=similar_examples,
                                               expected_return_tokens=get_predicted_token_usage,
                                               throttling=self.throttling)]  # student 1

        if get_feedback:
            student_first_answer, _ = answer_to_data(answers[0])
            teachers_feedback = self.teacher.give_feedback(raw_data=raw_data,
                                                           student_reply=student_first_answer,
                                                           throttling=self.throttling)

            final_answer = self._finalise_reply(raw_data=raw_data,
                                                initial_answer=student_first_answer.ner_tags,
                                                teacher_feedback=teachers_feedback.content,
                                                function_call_examples=similar_examples,
                                                throttling=self.throttling)

            answers.extend([teachers_feedback, final_answer])
        if save_db:
            db_prompt_client.insert_conversation(answers)
            self.logger.info(f"{raw_data['id']} stored in DB")

        return answers

    def _finalise_reply(self,
                        raw_data,
                        initial_answer,
                        teacher_feedback,
                        function_call_examples,
                        throttling: TokenThrottling = None):
        combined_prompt = f"""The Question Sentence is: {raw_data["tokens"]}
Your initial answer is: {initial_answer}
Peer Review: {teacher_feedback}
Please write your final answer. But you don't have to follow the Review if you're confident with your original answer."""
        final_reply = self.create_chat_with_agent(raw_question_data=raw_data,
                                                  user_sentence=combined_prompt,
                                                  function_call_examples=function_call_examples,
                                                  throttling=throttling)
        return final_reply


def get_similar_dataset(db_client, dataset, model_name, vector, data_id, num=2):
    from utils.main_utils import match_indices_from_base_dataset
    similar_keys = db_client.search_similar_vector_by_data_id(model_name, vector, num=num + 1)
    if similar_keys is None:
        logging.info("There is no similar data")
        return None
    data_ids = [key.split(f"{model_name}:")[-1] for key in similar_keys if key.split(":")[-1] != data_id.split(":")[-1]]
    # 자기 자신은 없지만 num개가 나올 때
    data_ids = data_ids[:num] if len(data_ids) > num else data_ids
    result: Dataset = match_indices_from_base_dataset(dataset, data_ids, remove=False)
    return result.select([i for i in range(num)])


def count_expected_tokens(tokens: list, return_token_count: bool = True):
    ner_tags = ["XYZ"] * len(tokens)
    dummy_example = {"response":
                         [{"word": token, "entity": tag} for token, tag in zip(tokens, ner_tags)]
                     }
    if return_token_count:
        return count_message_tokens([MessageFunctionCall(role="function", name="find_ner", content=dummy_example)])
    return MessageFunctionCall(role="function", name="find_ner", content=dummy_example)


def get_example(dataset, id_to_label=None, function_name="find_ner"):
    from functools import partial

    examples = []
    encode_fn = partial(make_example, id_to_label=id_to_label)
    entities = dataset.map(encode_fn)
    for entity_ex in entities:
        sample = entity_ex["response"]
        example = MessageFunctionCall(role="function",
                                      name=function_name,
                                      content={"response": sample})
        examples.append(example)

    return examples


def make_example(example, id_to_label=None):
    tokens = example['tokens']
    ner_tags = example["ner_tags"]

    if id_to_label is not None:
        return {"response": [{"word": token, "entity": id_to_label[tag]} for token, tag in zip(tokens, ner_tags)]}
    return {"response": [{"word": token, "entity": tag} for token, tag in zip(tokens, ner_tags)]}


def answer_to_data(answer, label_to_id=None, data_id=None):
    logger_name = f"{answer_to_data.__name__}"
    if data_id:
        logger_name += f"-{data_id}"

    logger = logging.getLogger(logger_name)
    fail_result = None, []
    retry_datum = fail_result[1]

    json_response = {"response": [{"entity": "", "word": ""}]}
    if data_id:
        json_response = json.loads(answer)

    else:
        data_id = answer.data_id
        if answer.function_call:
            json_response = json.loads(answer.function_call.arguments)
    try:
        response = json_response["response"]
        try:
            temp_entity = [i["entity"] for i in response]
            temp_word = [i["word"] for i in response]
        except KeyError as key_error:
            logger.error(f"KeyError in {data_id}: {key_error}")
            retry_datum.append(data_id)
            return fail_result
        except ValueError as value_error:
            logger.error(f"ValueError in {data_id}: {value_error}")
            retry_datum.append(data_id)
            return fail_result

    except KeyError as key_error:
        logger.error(f"KeyError in {data_id}: {key_error}")
        return fail_result

    if label_to_id:
        entities, words = zip(*[
            (label_to_id[entity_dict['entity']] if entity_dict['entity'] in label_to_id else label_to_id["O"],
             entity_dict['word'])
            for entity_dict in response
        ])
        return EntityAgentResponse(data_id=data_id, tokens=list(words), ner_tags=list(entities)), retry_datum

    entities, words = zip(*[(entity_dict['entity'], entity_dict['word']) for entity_dict in response])
    return EntityAgentResponse(data_id=data_id, tokens=list(words), ner_tags=list(entities)), retry_datum
