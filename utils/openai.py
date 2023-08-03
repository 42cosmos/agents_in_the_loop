import os
import time

import tiktoken
import logging
import logging.config
from dotenv import load_dotenv

import openai


# 60 parallel api calls without problem
class OpenAIGpt:
    def __init__(self, env_path=None):
        if env_path is not None:
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        self.logger = logging.getLogger("openai")
        self.__api_key = os.getenv("OPENAI_API_KEY")
        self.__organisation_key = os.getenv("ORGANISATION_KEY")
        assert self.__api_key is not None, "Please set OPENAI_API_KEY in .env file"

        if self.__organisation_key:
            openai.organization = self.__organisation_key

        openai.api_key = self.__api_key

    @property
    def api_key(self):
        raise Exception("Direct Access to api_key is not allowed")

    @api_key.setter
    def api_key(self, value):
        raise Exception("Direct modification of api_key is not allowed")

    def request(self, request_data: list, model: str, functions=None):
        request_data = [{"role": role, "content": content} for role, content in request_data]
        completion = openai.ChatCompletion.create(
            model=model,
            messages=request_data,
        )
        try:
            return completion['choices'][0]['message']['content'], int(completion["usage"]["total_tokens"])

        except openai.error.APIError as e:
            self.logger.exception(f"API Error: {e}")
            self.logger.exception(f"Status Code: {e.status_code}, Response: {e.response.json()}")
            return False, 0

        except openai.error.RateLimitError as rate_limit_e:
            self.logger.exception(f"Rate Limit Error: {rate_limit_e}")
            time.sleep(60)
            return False, 0

        except Exception as e:
            self.logger.exception(f"Error: {e}")
            return False, 0

    @staticmethod
    def num_tokens_from_messages(messages: str, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""

        msg_result = [{"role": role, "content": content} for role, content in messages]

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-4" or model == "gpt-4-0314":
            print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            model = "gpt-4-0314"
            tokens_per_message = 3
            tokens_per_name = 1

        elif model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted

        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

        num_tokens = 0
        for message in msg_result:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
