import time
import logging
from typing import Optional, List
from dataclasses import dataclass

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
            max_tokens=4096,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-0613",
            prompt_token_cost=0.0015,
            completion_token_cost=0.002,
            max_tokens=4096,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-16k-0613",
            prompt_token_cost=0.003,
            completion_token_cost=0.004,
            max_tokens=16384,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-4-0314",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-4-0613",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-4-32k-0314",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
            temperature=1,
        ),
        ChatModelInfo(
            name="gpt-4-32k-0613",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
            temperature=1,
        ),
    ]
}


@dataclass
class OpenAIFunctionCall:
    """Represents a function call as generated by an OpenAI model

    Attributes:
        name: the name of the function that the LLM wants to call
        arguments: a stringified JSON object (unverified) containing `arg: value` pairs
    """

    name: str
    arguments: str


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
    parameters: dict[str, ParameterSpec]

    @property
    def __dict__(self):
        """Output an OpenAI-consumable function specification"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                    }
                    for param in self.parameters.values()
                },
                "required": [
                    param.name for param in self.parameters.values() if param.required
                ],
            },
        }


# def get_openai_command_specs(agent: Agent) -> list[OpenAIFunctionSpec]:
#     """Get OpenAI-consumable function specs for the agent's available commands.
#     see https://platform.openai.com/docs/guides/gpt/function-calling
#     """
#     if not agent.config.openai_functions:
#         return []
#
#     return [
#         OpenAIFunctionSpec(
#             name=command.name,
#             description=command.description,
#             parameters={
#                 param.name: ParameterSpec(
#                     name=param.name,
#                     type=param.type,
#                     required=param.required,
#                     description=param.description,
#                 )
#                 for param in command.parameters
#             },
#         )
#         for command in agent.command_registry.commands.values()
#     ]


def retry_api(
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    error_messages = {
        ServiceUnavailableError: f"Error: The OpenAI API engine is currently overloaded, passing...",
        RateLimitError: f"Error: Reached rate limit, passing...",
    }
    api_key_error_msg = (
        f"Please double check that you have setup a PAID OpenAI API Account."
    )
    backoff_msg = (
        f"Error: API Bad gateway. Waiting {{backoff}} seconds..."
    )

    def _wrapper(func):
        import functools
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except (RateLimitError, ServiceUnavailableError) as e:
                    if attempt == num_attempts:
                        raise

                    error_msg = error_messages[type(e)]
                    logger.debug(error_msg)
                    if not user_warned:
                        logger.double_check(api_key_error_msg)
                        user_warned = True

                except (APIError, Timeout) as e:
                    if (e.http_status not in [429, 502]) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.debug(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

        return _wrapped

    return _wrapper


# @meter_api
@retry_api()
def create_chat_completion(
        messages: List[MessageDict],
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
            "entities": {
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
    "required": ["entities"]
}
