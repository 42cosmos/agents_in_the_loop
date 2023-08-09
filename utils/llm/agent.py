from __future__ import annotations

import json
import os
import copy
import logging
import logging.config
from dotenv import load_dotenv
from dataclasses import dataclass, field

import openai

from .chat import chat_with_agent
from .base import Message, MessageRole, MessageType
from .token_counter import count_string_tokens
from .openai import OPEN_AI_CHAT_MODELS

logger = logging.getLogger(f"{__name__}")

@dataclass
class MessageHistory:
    agent: Agent
    messages: list[Message] = field(default_factory=list)
    summary: str = "I was created"

    last_trimmed_index: int = 0

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
            type: MessageType | None = None,
    ):
        return self.append(Message(role, content, type))

    def append(self, message: Message):
        return self.messages.append(message)

    def trim_messages(
            self, current_message_chain: list[Message], config
    ) -> tuple[Message, list[Message]]:
        """
        Returns a list of trimmed messages: messages which are in the message history
        but not in current_message_chain.

        Args:
            current_message_chain (list[Message]): The messages currently in the context.
            config (Config): The config to use.

        Returns:
            Message: A message with the new running summary after adding the trimmed messages.
            list[Message]: A list of messages that are in full_message_history with an index higher than last_trimmed_index and absent from current_message_chain.
        """
        # Select messages in full_message_history with an index higher than last_trimmed_index
        new_messages = [
            msg for i, msg in enumerate(self) if i > self.last_trimmed_index
        ]

        # Remove messages that are already present in current_message_chain
        new_messages_not_in_chain = [
            msg for msg in new_messages if msg not in current_message_chain
        ]

        if not new_messages_not_in_chain:
            return self.summary_message(), []

        new_summary_message = self.update_running_summary(
            new_events=new_messages_not_in_chain, config=config
        )

        # Find the index of the last message processed
        last_message = new_messages_not_in_chain[-1]
        self.last_trimmed_index = self.messages.index(last_message)

        return new_summary_message, new_messages_not_in_chain

    def per_cycle(self, config, messages: list[Message] | None = None):
        """
        Yields:
            Message: a message containing user input
            Message: a message from the AI containing a proposed action
            Message: the message containing the result of the AI's proposed action
        """
        messages = messages or self.messages
        for i in range(0, len(messages) - 1):
            ai_message = messages[i]
            if ai_message.type != "ai_response":
                continue
            user_message = (
                messages[i - 1] if i > 0 and messages[i - 1].role == "user" else None
            )
            result_message = messages[i + 1]
            try:
                assert (
                        extract_json_from_response(ai_message.content) != {}
                ), "AI response is not a valid JSON object"
                assert result_message.type == "action_result"

                yield user_message, ai_message, result_message
            except AssertionError as err:
                logger.debug(
                    f"Invalid item in message history: {err}; Messages: {messages[i - 1:i + 2]}"
                )

    def summary_message(self) -> Message:
        return Message(
            "system",
            f"This reminds you of these events from your past: \n{self.summary}",
        )

    def update_running_summary(
            self, new_events: list[Message], config
    ) -> Message:
        """
        This function takes a list of dictionaries representing new events and combines them with the current summary,
        focusing on key and potentially important information to remember. The updated summary is returned in a message
        formatted in the 1st person past tense.

        Args:
            new_events (List[Dict]): A list of dictionaries containing the latest events to be added to the summary.

        Returns:
            str: A message containing the updated summary of actions, formatted in the 1st person past tense.

        Example:
            new_events = [{"event": "entered the kitchen."}, {"event": "found a scrawled note with the number 7"}]
            update_running_summary(new_events)
            # Returns: "This reminds you of these events from your past: \nI entered the kitchen and found a scrawled note saying 7."
        """
        if not new_events:
            return self.summary_message()

        # Create a copy of the new_events list to prevent modifying the original list
        new_events = copy.deepcopy(new_events)

        # Replace "assistant" with "you". This produces much better first person past tense results.
        for event in new_events:
            if event.role.lower() == "assistant":
                event.role = "you"

                # Remove "thoughts" dictionary from "content"
                try:
                    content_dict = extract_json_from_response(event.content)
                    if "thoughts" in content_dict:
                        del content_dict["thoughts"]
                    event.content = json.dumps(content_dict)
                except json.JSONDecodeError as e:
                    logger.error(f"Error: Invalid JSON: {e}")
                    if config.debug_mode:
                        logger.error(f"{event.content}")

            elif event.role.lower() == "system":
                event.role = "your computer"

            # Delete all user messages
            elif event.role == "user":
                new_events.remove(event)

        # Summarize events and current summary in batch to a new running summary

        # Assume an upper bound length for the summary prompt template, i.e. Your task is to create a concise running summary...., in summarize_batch func
        prompt_template_length = 100
        max_tokens = OPEN_AI_CHAT_MODELS.get(config.fast_llm_model).max_tokens
        summary_tlength = count_string_tokens(str(self.summary), config.fast_llm_model)
        batch = []
        batch_tlength = 0

        for event in new_events:
            event_tlength = count_string_tokens(str(event), config.fast_llm_model)

            if (
                    batch_tlength + event_tlength
                    > max_tokens - prompt_template_length - summary_tlength
            ):
                # The batch is full. Summarize it and start a new one.
                self.summarize_batch(batch, config)
                summary_tlength = count_string_tokens(
                    str(self.summary), config.fast_llm_model
                )
                batch = [event]
                batch_tlength = event_tlength
            else:
                batch.append(event)
                batch_tlength += event_tlength

        if batch:
            # There's an unprocessed batch. Summarize it.
            self.summarize_batch(batch, config)

        return self.summary_message()

    def summarize_batch(self, new_events_batch, config):
        pass
#         prompt = f'''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.
#
# You will receive the current summary and your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.
#
# Summary So Far:
# """
# {self.summary}
# """
#
# Latest Development:
# """
# {new_events_batch or "Nothing new happened."}
# """
# '''
#
#         prompt = ChatSequence.for_model(
#             config.fast_llm_model, [Message("user", prompt)]
#         )
#         self.agent.log_cycle_handler.log_cycle(
#             self.agent.ai_name,
#             self.agent.created_at,
#             self.agent.cycle_count,
#             prompt.raw(),
#             PROMPT_SUMMARY_FILE_NAME,
#         )
#
#         self.summary = create_chat_completion(prompt, config).content
#
#         self.agent.log_cycle_handler.log_cycle(
#             self.agent.ai_name,
#             self.agent.created_at,
#             self.agent.cycle_count,
#             self.summary,
#             SUMMARY_FILE_NAME,
#         )


class Agent:
    def __init__(self, role, config, db_client=None):
        self.role = role
        self.raw_config = config
        self.config = config[role]
        self.db_client = db_client

        load_dotenv(dotenv_path=self.config.env_path) if self.config.env_path is not None else load_dotenv()
        self.max_conversation_limit = self.config.max_conversation_limit
        self.history = MessageHistory(self)
        self.system_prompt = None
        self.logger = logging.getLogger("openai")
        self.logger.info(f"{role} Agent is Ready to talk ! ")

        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "Please set OPENAI_API_KEY in .env file"

        self.organisation_key = os.getenv("ORGANISATION_KEY")

        openai.api_key = self.api_key
        if self.organisation_key:
            openai.organization = self.organisation_key

    def _get_reply(self, prompt):
        assert self.system_prompt is not None, f"Please set system_prompt for {self.role} agent"
        agent_reply = self.create_chat_with_agent(prompt)
        return agent_reply

    def create_chat_with_agent(self, prompt):
        assistant_reply = chat_with_agent(
            agent=self,
            system_prompt=self.system_prompt,
            triggering_prompt=prompt
        )
        return assistant_reply

    def get_memories(self):
        return self.db_client.get_all_from_list(self.config.memory_key)


class Teacher(Agent):
    def __init__(self, config, db_client=None):
        super().__init__("teacher", config, db_client)
        self.system_prompt = "You are a maths professor at a university. You need to find out what is wrong with a student's answer and point it out to them, but you must never tell them the correct answer. You need to give them instructions to help them get the right answer."

    def give_feedback(self, question, student_reply):
        prompt = f"""The Question is "{question}"\n Student: {student_reply}"""
        assistant_reply = self.create_chat_with_agent(prompt)
        logging.info("Teacher Feedback is given...")
        return assistant_reply


class Student(Agent):
    def __init__(self, config, db_client=None):
        super().__init__("student", config, db_client)
        self.system_prompt = "You are university student major in Mathematics. You are talking to your teacher."
        self.teacher = Teacher(config=self.raw_config)

    def talk_to_agent(self, prompt):
        initial_answer = self.create_chat_with_agent(prompt)
        feedback = self.teacher.give_feedback(initial_answer, prompt)
        final_answer = self._finalise_reply(prompt, initial_answer, feedback)
        return final_answer

    def _finalise_reply(self, question, initial_reply, teacher_feedback):
        combined_prompt = f"This Problem ({question})'s your initial answer is: {initial_reply}\n Teacher's feedback: {teacher_feedback}\n Please write your final answer. but you have to write only answer not explanation."
        final_reply = self._get_reply(combined_prompt)
        return final_reply
