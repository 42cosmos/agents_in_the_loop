from __future__ import annotations

import logging
from typing import Optional, List
from colorama import Fore

from .base import (
    Message,
    ChatSequence,
    ChatModelResponse,
    MessageFunctionCall,
)

from .openai import (
    OpenAIFunctionSpec,
    OPEN_AI_CHAT_MODELS,
    ner_gpt_function,
    create_chat_completion as openai_chat_completion
)

from .token_counter import count_message_tokens

logger = logging.getLogger(__name__)


def create_chat_completion(
        agent,
        prompt: ChatSequence,
        used_tokens: Optional[int],
        functions: Optional[List[OpenAIFunctionSpec]] = None,
        reserved_tokens: Optional[int] = 0,
) -> ChatModelResponse:
    config = agent.config
    model = OPEN_AI_CHAT_MODELS[config.model_name]
    model_max_tokens = model.max_tokens

    reserved_tokens += 1000  # 1000 tokens for safety

    # 넘어온 used_tokens은 사용자가 입력한 토큰 길이
    if used_tokens is None:
        model_max_tokens -= prompt.token_length
    else:
        model_max_tokens -= used_tokens

    # 다 더해서 4096 미만이어야 함
    # 대답을 위한 토큰이 부족할 경우
    if model_max_tokens - reserved_tokens <= 0:
        model_max_tokens += int(reserved_tokens / 2)

    logging.info(
        f"used_tokens:: {count_message_tokens(prompt)} - model_max_tokens:: {model_max_tokens} - reserved_tokens:: {reserved_tokens}")

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, max_tokens {model_max_tokens}{Fore.RESET}"
    )

    chat_completion_kwargs = {"model": config.model_name,
                              "max_tokens": model_max_tokens}

    if functions:
        chat_completion_kwargs["functions"] = functions
        chat_completion_kwargs["function_call"] = {"name": functions[0]["name"]}

        for function in functions:
            logger.debug(f"Function dicts: {function['name']}")

    response = openai_chat_completion(
        role=agent.role,
        messages=prompt.raw(),
        **chat_completion_kwargs,
    )
    if not response:
        return ChatModelResponse(
            model_info=model,
            content=None,
            function_call=None,
        )

    if hasattr(response, "error"):
        logger.error(response.error)
        raise RuntimeError(response.error)

    first_message = response.choices[0].message
    content: str | None = first_message.get("content")
    function_call = first_message.get("function_call")

    return ChatModelResponse(
        model_info=model,
        content=content,
        function_call=function_call,
    )


def chat_with_agent(agent,
                    system_prompt: str,
                    llm_guidance: str,
                    triggering_prompt: str,
                    use_functions: bool = True,
                    function_call_examples: List[MessageFunctionCall] = None,
                    expected_return_tokens: int = 0,
                    ):
    config = agent.config
    model = config.model_name
    role = agent.role

    assert model is not None, "model must be specified"

    message_sequence = ChatSequence.for_model(
        model,
        [
            Message("system", system_prompt),
        ],
    )

    if llm_guidance is not None:
        llm_guide_msg = Message("user", llm_guidance)
        message_sequence.append(llm_guide_msg)

    functions = []
    if role == "student":
        functions.append(ner_gpt_function)
    # elif role == "teacher":
    #     functions.append(teacher_response_function.to_dict())

    if not use_functions and function_call_examples:
        raise "You can not use functions without function_call_examples"

    if use_functions and function_call_examples:
        for example in function_call_examples:
            message_sequence.append(example)

    user_input_msg = Message("user", triggering_prompt)
    message_sequence.append(user_input_msg)

    send_token_limit = count_message_tokens(message_sequence.messages)
    send_token_limit += 60  # 53 in the functions --> InvalidRequestError
    # However, you requested 4150 tokens (649 in the messages, 53 in the functions, and 3448 in the completion)

    assistant_reply = create_chat_completion(
        agent=agent,
        prompt=message_sequence,
        functions=functions,
        used_tokens=send_token_limit,
        reserved_tokens=expected_return_tokens
    )

    # Update full message history
    agent.history.append(user_input_msg)
    agent.history.add("assistant", assistant_reply, assistant_reply.function_call)

    return assistant_reply
