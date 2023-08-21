from __future__ import annotations

import json
import os
import logging
import logging.config
from json.decoder import JSONDecodeError
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
    ChatModelResponse,
    OpenAIFunctionCall,
    MessageFunctionCall,
    FeedbackAgentResponse, EntityAgentResponse
)

from .token_counter import count_message_tokens

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

        assert dataset_lang in LANGUAGES, f"dataset_lang should be one of {LANGUAGES.keys()}"

        self.api_cost_accumulated = 0.0

        load_dotenv(dotenv_path=self.config.env_path) if self.config.env_path is not None else load_dotenv()
        self.max_conversation_limit = self.config.max_conversation_limit
        self.history = MessageHistory(self)

        system_prompt_key_for_redis = f"{self.role}:base_system"
        self.system_prompt: str = self.db_client.get_prompt(data_id=system_prompt_key_for_redis,
                                                            doc_prefix="prompt")

        self.logger = logging.getLogger("openai")
        if self.system_prompt is None:
            raise ValueError(f"Please set system_prompt for {self.role} agent")

        self.logger.info(f"{Fore.YELLOW}{role.upper()} Agent{Fore.RESET} is Ready to talk !")

        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "Please set OPENAI_API_KEY in .env file"
        openai.api_key = self.api_key

        self.organisation_key = os.getenv("ORGANISATION_KEY")
        if self.organisation_key:
            openai.organization = self.organisation_key

    def _get_basic_prompt(self):
        information = self.db_client.get_prompt(data_id=f"{self.dataset_name}:{LANGUAGES[self.dataset_lang]}",
                                                doc_prefix="prompt")
        warning_sign = ""
        if self.role == "student":
            warning_sign = self.db_client.get_prompt(data_id=f"warning",
                                                     doc_prefix="prompt")

        elif self.role == "teacher":
            warning_sign = self.db_client.get_prompt(data_id=f"teacher:warning",
                                                     doc_prefix="prompt")

        return f"{information}\n{warning_sign}"

    # def get_cost(self, data_id, answers: List[ChatModelResponse]):
    #     cost = 0.0
    #     for answer in answers:
    #         # input_cost_per_tokens = answer.model_info.prompt_token_cost
    #         # output_cost_per_tokens = answer.model_info.completion_token_cost
    #
    #         # cost += api_cost
    #         # self.api_cost_accumulated += api_cost
    #     logging.info(f"Total cost of {data_id} conversation is {self.api_cost_accumulated} tokens")

    def _get_prompt(self, triggering_prompt):
        return [Message("system", self.system_prompt),
                Message("user", self.ner_guidance),
                Message("user", triggering_prompt)]

    def create_chat_with_agent(self,
                               user_sentence,
                               function_call_examples=None,
                               expected_return_tokens: int = 0):
        assert user_sentence, "Please set your question"

        if function_call_examples is not None:
            assert isinstance(function_call_examples, list)
            assert all([isinstance(example, MessageFunctionCall) for example in function_call_examples])

        assistant_reply = chat_with_agent(
            agent=self,
            system_prompt=self.system_prompt,
            llm_guidance=self.ner_guidance,
            triggering_prompt=user_sentence,
            function_call_examples=function_call_examples,
            expected_return_tokens=expected_return_tokens
        )
        return assistant_reply


class Teacher(Agent):
    def __init__(self, config, db_client, dataset_name, dataset_lang):
        super().__init__("teacher", config, db_client, dataset_name, dataset_lang)

    def give_feedback(self, student_reply, problem_sentence: list):
        if student_reply is None:
            prompt = f"""There is no answer given by the student.\n Question Sentence: {str(problem_sentence)}"""

        else:
            problem_sentence = student_reply.tokens
            student_answer = student_reply.ner_tags
            prompt = f"""Here's the sentence in which the student has to find the entity, and the answer given by the student:\nQuestion Sentence: {str(problem_sentence)}\nStudent's Answer: "{student_answer}" """.strip()
        assistant_reply = self.create_chat_with_agent(prompt)

        return assistant_reply


class Student(Agent):
    def __init__(self, config, db_client, dataset_name, dataset_lang):
        super().__init__("student", config, db_client, dataset_name, dataset_lang)
        self.teacher = Teacher(config=self.raw_config,
                               db_client=db_client,
                               dataset_name=dataset_name,
                               dataset_lang=dataset_lang)

    def talk_to_agent(self,
                      data_id,
                      request_tokens: list,
                      similar_example: List[MessageFunctionCall],
                      get_feedback=False,
                      save_db=True):

        prompt = str(request_tokens)
        get_predicted_token_usage = count_expected_tokens(request_tokens)

        answers = [self.create_chat_with_agent(prompt,
                                               function_call_examples=similar_example,
                                               expected_return_tokens=get_predicted_token_usage)]  # student 1

        if get_feedback:
            student_first_answer = answer_to_data(data_id, answers[0])
            teachers_feedback = self.teacher.give_feedback(student_first_answer,
                                                           problem_sentence=request_tokens)

            final_answer = self._finalise_reply(question_sentence=request_tokens,
                                                initial_answer=student_first_answer.ner_tags,
                                                teacher_feedback=teachers_feedback.content,
                                                function_call_examples=similar_example)

            answers.extend([teachers_feedback, final_answer])

        # self.get_cost(answers)

        if save_db:
            self.db_client.insert_conversation(prompt_datum=answers, data_id=data_id)

        return answers

    def _finalise_reply(self, question_sentence, initial_answer, teacher_feedback, function_call_examples):
        combined_prompt = f"""The Question Sentence is: {question_sentence}
Your initial answer is: {initial_answer}
Peer Review: {teacher_feedback}
Please write your final answer. But you don't have to follow the Review if you're confident with your original answer."""
        final_reply = self.create_chat_with_agent(combined_prompt, function_call_examples=function_call_examples)
        return final_reply

    # def total_api_cost(self):
    #     return self.api_cost_accumulated + self.teacher.api_cost_accumulated


def get_similar_dataset(db_client, dataset, model_name, vector, data_id, num=2):
    from utils.main_utils import match_indices_from_base_dataset
    similar_keys = db_client.search_similar_vector_by_data_id(model_name, vector, num=num + 1)
    if similar_keys is None:
        logging.info("There is no similar data")
        return None
    data_ids = [key.split(f"{model_name}:")[-1] for key in similar_keys if key.split(":")[-1] != data_id.split(":")[-1]]
    # 자기 자신은 없지만 3개가 나올 때
    data_ids = data_ids[:num + 1] if len(data_ids) > num else data_ids
    result: Dataset = match_indices_from_base_dataset(dataset, data_ids, remove=False)
    return result.select([i for i in range(num)])


def count_expected_tokens(tokens: list):
    ner_tags = ["XYZ"] * len(tokens)
    dummy_example = {"response":
                         [{"word": token, "entity": tag} for token, tag in zip(tokens, ner_tags)]
                     }
    return count_message_tokens([MessageFunctionCall(role="function",
                                                     name="find_ner",
                                                     content=dummy_example)])


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


def answer_to_data(data_id, answer):
    arguments_return = answer.function_call.arguments
    retval_json = json.loads(arguments_return)
    # TODO: keyError 시 대응 방안
    try:
        response = retval_json["response"]
    except KeyError as key_error:
        logging.error(f"KeyError in {data_id}: {key_error}")
        return {"response": [
            {"entity": "", "word": ""}
        ]}

    entities, words = zip(*[(entity_dict['entity'], entity_dict['word'])
                            for entity_dict in response])
    return EntityAgentResponse(data_id=data_id, tokens=list(words), ner_tags=list(entities))
