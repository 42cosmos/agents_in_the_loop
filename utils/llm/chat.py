from __future__ import annotations

import time
import logging
from typing import Optional, List
from easydict import EasyDict

from .base import Message, ChatSequence, ChatModelResponse
from .openai import OpenAIFunctionSpec, OPEN_AI_CHAT_MODELS, create_chat_completion as openai_chat_completion

logger = logging.getLogger(__name__)


def create_chat_completion(
        agent,
        prompt: ChatSequence,
        functions: Optional[List[OpenAIFunctionSpec]] = None,
) -> ChatModelResponse:
    """Create a chat completion using the OpenAI API

    Returns:
        str: The response from the chat completion
    """

    # logger.debug(
    #     f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    # )
    config = agent.config
    model = OPEN_AI_CHAT_MODELS[config.model_name]
    temperature = 1
    max_tokens = model.max_tokens

    chat_completion_kwargs = {"model": config.model_name, "temperature": temperature,
                              }

    if functions:
        chat_completion_kwargs["functions"] = [
            function.__dict__ for function in functions
        ]
        logger.debug(f"Function dicts: {chat_completion_kwargs['functions']}")

    response = openai_chat_completion(
        messages=prompt.raw(),
        **chat_completion_kwargs,
    )
    logger.info(f"Response: {response}")

    if hasattr(response, "error"):
        logger.error(response.error)
        raise RuntimeError(response.error)

    first_message = response.choices[0].message
    content: str | None = first_message.get("content")
    # function_call: OpenAIFunctionCall | None = first_message.get("function_call")

    return ChatModelResponse(
        model_info=model,
        content=content,
        # function_call=function_call,
    )


def chat_with_agent(agent,
                    system_prompt: str,
                    triggering_prompt: str,
                    ):
    config = agent.config
    model = config.model_name

    assert model is not None, "model must be specified"

    # Reserve 1000 tokens for the response
    # logger.debug(f"Token limit: {token_limit}")
    # send_token_limit = token_limit - 1000

    message_sequence = ChatSequence.for_model(
        model,
        [
            # 최초 프롬프트
            Message("system", system_prompt),
        ],
    )

    user_input_msg = Message("user", triggering_prompt)

    message_sequence.append(user_input_msg)

    assistant_reply = create_chat_completion(
        agent=agent,
        prompt=message_sequence,
        # functions=get_openai_command_specs(agent),
        # max_tokens=tokens_remaining,
    )
    # Update full message history
    agent.history.append(user_input_msg)
    agent.history.add("assistant", assistant_reply.content)

    return assistant_reply