import ast
import functools
import time
import logging
from typing import Optional, List
from dataclasses import dataclass

from colorama import Fore, Style

from .base import ChatModelInfo, MessageDict

from openai.openai_object import OpenAIObject
import openai.api_resources.abstract.engine_api_resource as engine_api_resource
from openai.error import APIError, RateLimitError, ServiceUnavailableError, Timeout

logger = logging.getLogger(__name__)

OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name="gpt-3.5-turbo-0301",
            prompt_token_cost=0.0015,
            completion_token_cost=0.002,
            max_tokens=4096
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-0613",
            prompt_token_cost=0.0015,
            completion_token_cost=0.002,
            max_tokens=4096,
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-16k-0613",
            prompt_token_cost=0.003,
            completion_token_cost=0.004,
            max_tokens=16384,
        ),
        ChatModelInfo(
            name="gpt-4-0314",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
        ),
        ChatModelInfo(
            name="gpt-4-0613",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
        ),
        ChatModelInfo(
            name="gpt-4-32k-0314",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
        ),
        ChatModelInfo(
            name="gpt-4-32k-0613",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
        ),
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
        num_retries: int = 2,
        backoff_base: float = 2.0,
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
            user_warned = not warn_user
            for attempt in range(1, num_retries + 1):
                try:
                    return func(*args, **kwargs)

                except (JSONDecodeError, RateLimitError, ServiceUnavailableError, TokenMismatchError) as e:
                    if attempt == num_retries:
                        if isinstance(e, (JSONDecodeError, TokenMismatchError)):
                            logger.error(f"Max retries reached. Returning empty value due to {type(e).__name__}.")
                            return False
                        else:
                            raise

                    error_msg = error_messages[type(e)]
                    logger.debug(error_msg)

                    if isinstance(e, JSONDecodeError):
                        logger.error(f"{json_decode_error_msg} - attempt {attempt} of {num_retries}")
                        user_warned = True

                    if isinstance(e, TokenMismatchError):
                        logger.error(f"{error_msg} - attempt {attempt} of {num_retries}")
                        user_warned = True

                    if isinstance(e, RateLimitError):
                        logger.error(f"RateLimitError occurred. Waiting 60 seconds...")
                        time.sleep(60)
                        user_warned = True

                    if not user_warned and type(e) is not JSONDecodeError:
                        logger.error(api_key_error_msg)
                        user_warned = True

                except (APIError, Timeout) as e:
                    if (e.http_status not in [429, 502]) or (attempt == num_retries):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.debug(backoff_msg.format(backoff=backoff))
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
    completion = openai.ChatCompletion.create(
        messages=messages,
        **kwargs,
    )

    if role == "student":
        import json
        check_json_decode_error = json.loads(completion.choices[0].message.function_call["arguments"])

        if len(question_tokens) != len(check_json_decode_error["response"]):
            raise TokenMismatchError("Mismatch between question tokens and response tokens.")

    if not hasattr(completion, "error"):
        logger.debug(f"Response: {completion}")
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
