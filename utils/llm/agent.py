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
from redis.exceptions import DataError

from utils.llm.chat import chat_with_agent
from utils.llm.base import (
    Message,
    MessageRole,
    OpenAIFunctionCall,
    MessageFunctionCall,
    EntityAgentResponse, ChatSequence
)
from utils.llm.openai_utils import OPEN_AI_CHAT_MODELS

from utils.llm.token_counter import count_message_tokens
from utils.throttling import TokenThrottling

logger = logging.getLogger(f"{__name__}")
logger.setLevel(logging.DEBUG)

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
        self.logger.setLevel(logging.DEBUG)

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
        information = self.db_client.get_value(f"prompt:{self.dataset_name}")
        assert information is not None, f"Please set prompt:{self.dataset_name} in redis"

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
            ],
        )

        if examples:
            for example in examples:
                message_sequence.append(example)

        message_sequence.append(Message("user", triggering_prompt))
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

    def give_feedback(self,
                      raw_data,
                      student_reply,
                      throttling: TokenThrottling = None,
                      raise_token_mismatch: bool = False):
        prompt = "You MUST TELL the student that they have to create an answer based on the truncated words in the given sentence. The Student MUST answer based on the truncated words removed from the given sentence."
        if raise_token_mismatch:
            self.logger.info(f"{raw_data['id']} has token mismatch in First Answer")
            prompt += f"""But The Student has changed the separated words in the sentence. Student should have answered {len(raw_data["tokens"])}, but answered {len(student_reply.tokens)}. EMPHASISE that student should only answer as many objects as there are truncated words in the given sentence."""
            prompt += f"and Please give some instructions to solve this question to find the entity.\nQuestion Sentence: {raw_data['tokens']}\nStudent's Answer: {student_reply.ner_tags} "

        elif student_reply is None or student_reply.tokens is None:
            self.logger.info(f"{raw_data['id']} has no answer")
            prompt = f"""There is no given answer from student. Please give some detailed instructions to solve this question to find the entity. \nQuestion Sentence: {raw_data["tokens"]}"""

        else:
            problem_sentence = student_reply.tokens
            student_answer = student_reply.ner_tags
            prompt = f"""Here's the sentence in which the student has to find the entity, and the answer given by the student:\nQuestion Sentence: {problem_sentence}\nStudent's Answer: "{student_answer}" """.strip()
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
        self.dataset_lang = dataset_lang
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
        get_predicted_token_usage = count_expected_tokens(raw_data["tokens"])

        self.logger.info(f"Processing example: {raw_data['id']}")
        original_sent = " ".join(raw_data["tokens"]) if not self.dataset_lang == "ja" else "".join(raw_data["tokens"])
        user_sentence = f"""Original Sentence: {original_sent}\nDivided as: {raw_data["tokens"]}"""
        answers = {"1": self.create_chat_with_agent(user_sentence=user_sentence,
                                                    raw_question_data=raw_data,
                                                    function_call_examples=similar_examples,
                                                    expected_return_tokens=get_predicted_token_usage,
                                                    throttling=self.throttling)}  # student 1

        if get_feedback:
            raise_token_mismatch = False
            if answers["1"].content:  # TokenMismatchError로 인해 값이 function_call에 들어가지 못하고 content에 안착
                self.logger.info(f"{raw_data['id']} has token mismatch in First Answer")
                raise_token_mismatch = True

            student_first_answer, _ = answer_to_data(answers["1"], raise_token_mismatch=raise_token_mismatch)
            teachers_feedback = self.teacher.give_feedback(raw_data=raw_data,
                                                           student_reply=student_first_answer,
                                                           throttling=self.throttling,
                                                           raise_token_mismatch=raise_token_mismatch)

            final_answer = self._finalise_reply(raw_data=raw_data,
                                                initial_answer=student_first_answer,
                                                teacher_feedback=teachers_feedback.content,
                                                function_call_examples=similar_examples,
                                                throttling=self.throttling)

            answers["2"] = teachers_feedback
            answers["3"] = final_answer

        if save_db:
            if db_prompt_client:
                try:
                    db_prompt_client.insert_conversation(answers)
                    self.logger.info(f"{raw_data['id']} stored in DB")
                except DataError as data_error:
                    self.logger.error(f"{answers}\n\n{raw_data['id']} has an error in DB: {data_error}")
                    return []

        return answers

    def _finalise_reply(self,
                        raw_data,
                        initial_answer,
                        teacher_feedback,
                        function_call_examples,
                        throttling: TokenThrottling = None):

        combined_prompt = f"""Peer Review: {teacher_feedback}\n\nPlease write your final answer."""
        combined_prompt += f"""Question Sentence: {raw_data["tokens"]}\nYour initial answer: """
        if initial_answer is None or initial_answer.ner_tags is None:
            combined_prompt += "Nothing"
        else:
            combined_prompt += f"{initial_answer.ner_tags}"

        combined_prompt += f""" you should have answered {len(raw_data["tokens"])}. You have to answer as many things as there are truncated words in the given sentence."""

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


def answer_to_data(answer, label_to_id=None, data_id=None, raise_token_mismatch: bool = False):
    logger_name = f"{answer_to_data.__name__}"
    if data_id:
        # data_id -> DB의 값을 가져오는 경우 값 존재
        logger_name += f"-{data_id}"

    logger = logging.getLogger(logger_name)
    fail_result = None, []
    retry_datum = fail_result[1]

    json_response = {"response": []}
    if data_id:
        try:
            json_response = json.loads(answer)
        except Exception as e:
            logger.warning(f"Exception in {data_id}: {answer}")
            logger.warning(f"Exception in {data_id}: {e}")

    else:
        data_id = answer.data_id
        answer = answer.content if raise_token_mismatch else answer.function_call

        if answer:
            json_response = json.loads(answer.arguments)

    response = json_response["response"]
    if response:
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

        except TypeError as type_error:
            # LLM이 word와 entity 를 반대로 작성하는 경우가 종종 있음
            logger.error(f"TypeError in {data_id}: {type_error}")
            retry_datum.append(data_id)
            return fail_result
    else:
        retry_datum.append(data_id)
        return fail_result

    if label_to_id:
        entities, words = zip(*[
            (label_to_id[entity_dict['entity']] if entity_dict['entity'] in label_to_id else label_to_id["O"],
             entity_dict['word'])
            for entity_dict in response
        ])
        result = EntityAgentResponse(data_id=data_id, tokens=list(words), ner_tags=list(entities)), retry_datum
    else:
        entities, words = zip(*[(entity_dict['entity'], entity_dict['word']) for entity_dict in response])
        result = EntityAgentResponse(data_id=data_id, tokens=list(words), ner_tags=list(entities)), retry_datum

    if result[0].tokens and result[0].ner_tags:
        return result
    else:
        retry_datum.append(data_id)
        return fail_result
