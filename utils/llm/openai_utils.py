import time
import logging
import functools
from typing import Optional, List
from dataclasses import dataclass

from colorama import Fore, Style

from utils.llm.base import ChatModelInfo, MessageDict
from openai.error import APIError, RateLimitError, ServiceUnavailableError, Timeout, InvalidRequestError

logging.basicConfig(level=logging.DEBUG)

OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name="gpt-3.5-turbo-0301",
            prompt_token_cost=0.0015,
            completion_token_cost=0.002,
            max_tokens=4096,
            TPM=500000,
            RPM=6000
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-0613",
            prompt_token_cost=0.0015,
            completion_token_cost=0.002,
            max_tokens=4096,
            TPM=500000,
            RPM=6000
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-16k-0613",
            prompt_token_cost=0.003,
            completion_token_cost=0.004,
            max_tokens=16384,
            TPM=180000,
            RPM=3500
        ),
        ChatModelInfo(
            name="gpt-4-0314",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            TPM=10000,
            RPM=200
        ),
        ChatModelInfo(
            name="gpt-4-0613",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            TPM=10000,
            RPM=200
        )
    ]
}


@dataclass
class ParameterSpec:
    name: str
    type: str
    description: Optional[str]
    required: bool = False


@dataclass
class OpenAIFunctionSpec:
    """Represents a "function" in OpenAI, which is mapped to a Command in Auto-GPT"""

    name: str
    description: str
    parameters: dict[str]

    def to_dict(self):
        return {"name": self.name,
                "description": self.description,
                "parameters": self.parameters}


class TokenMismatchError(Exception):
    """Raised when there's a mismatch between token length."""
    pass


def retry_api(
        num_retries: int = 5,
        warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    from json.decoder import JSONDecodeError
    error_messages: dict = {
        ServiceUnavailableError: f"{Fore.RED}Error: The OpenAI API engine is currently overloaded, passing...{Fore.RESET}",
        RateLimitError: f"{Fore.RED}Error: Reached rate limit, passing...{Fore.RESET}",
        JSONDecodeError: f"{Fore.RED}Error: Failed to decode JSON response, passing...{Fore.RESET}",
        TokenMismatchError: f"{Fore.RED}Error: Mismatch between question tokens and response tokens, passing...{Fore.RESET}",
        KeyError: f"{Fore.RED}Error: Key error, passing...{Fore.RESET}",
        InvalidRequestError: f"{Fore.RED}Error: Invalid request, passing...{Fore.RESET}",
    }

    json_decode_error_msg = (
        f"{Fore.RED}Error: Failed to decode JSON response. {Fore.RESET}"
    )

    api_key_error_msg = (
        f"Please double check that you have setup a "
        f"{Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account."
    )
    backoff_msg = (
        f"{Fore.RED}Error: API Bad gateway. Waiting {{backoff}} seconds...{Fore.RESET}"
    )

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            retry_logger = logging.getLogger(f"retry_api")
            user_warned = not warn_user
            for attempt in range(1, num_retries + 1):
                try:
                    return func(*args, **kwargs)

                except (InvalidRequestError, RateLimitError, ServiceUnavailableError,  # openai error
                        TokenMismatchError, KeyError, JSONDecodeError) as e:  # response error
                    error_msg = error_messages[type(e)]
                    retry_logger.error(f"{error_msg} - attempt {attempt} of {num_retries}")

                    if attempt == num_retries:
                        retry_logger.error(f"Max retries reached. Returning empty value due to {type(e).__name__}.")
                        return False

                    if isinstance(e, RateLimitError):
                        retry_logger.error(f"{error_msg} occurred. Waiting 60 seconds...")
                        time.sleep(30)

                    if isinstance(e, InvalidRequestError):
                        retry_logger.error(f"{error_msg} occurred. Skip all retries and return empty value.")
                        return False

                    if not user_warned:
                        if not isinstance(e, (TokenMismatchError, KeyError, JSONDecodeError)):
                            retry_logger.debug(f"Status: {e.http_status}")
                            retry_logger.debug(f"Response Body: {e.json_body}")
                            retry_logger.debug(f"Response Headers: {e.headers}")

                except (APIError, Timeout) as e:
                    if (e.http_status not in [429, 502]) or (attempt == num_retries):
                        retry_logger.error(f"{e} occurred. Max retries reached. Returning empty value.")
                        return False

                    elif e.http_status == 443:
                        user_warned = True
                        if attempt == num_retries:
                            retry_logger.error(f"{e} occurred. Max retries reached. Returning empty value.")
                            return False

                backoff = 60  # 1 minute
                retry_logger.warning(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

        return _wrapped

    return _wrapper


@retry_api()
def create_chat_completion(
        messages: List[MessageDict],
        question_tokens,
        role,
        *_,
        **kwargs,
):
    import openai
    """Create a chat completion using the OpenAI API

    Args:
        messages: A list of messages to feed to the chatbot.
        kwargs: Other arguments to pass to the OpenAI API chat completion call.
    Returns:
        OpenAIObject: The ChatCompletion response from OpenAI

    """

    chat_logger = logging.getLogger(f"create_chat_completion")

    completion = openai.ChatCompletion.create(
        messages=messages,
        **kwargs,
    )

    if role == "student":
        import json
        check_json_decode_error = json.loads(completion.choices[0].message.function_call.arguments)

        if len(question_tokens) != len(check_json_decode_error["response"]):
            raise TokenMismatchError("Mismatch between question tokens and response tokens.")

    if not hasattr(completion, "error"):
        chat_logger.warning(f"Response: {completion}")
    return completion


# temp variable
# TODO: remove this and use get_openai_command_specs above...
ner_gpt_function = {
    "name": "find_ner",
    "description": "Extracts named entities and their categories from the input text.",
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string",
                                 "description": "A word extracted from text."},
                        "entity": {"type": "string",
                                   "description": "Category of the named entity."}
                    }
                }
            }
        }
    },
    "required": ["response"]
}
