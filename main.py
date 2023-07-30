import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List

# Load your OpenAI API key
# models.OpenAI.api_key = process
# or from environment variable:
models.OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = """
You are an expert code reviewer. Your main job is to provide clear & very precise explanation of the code in not more than 200 words, identifying the exact line number. If and only if the user asks to review the code, only then will you provide suggestions to improve the code. Refrain from providing unnecessary comments. Your explanation & suggestions should be formatted in the following way:\n\n ☀️ Line {{line number}}: {{explanation}}
"""


@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history,
        model="gpt-3.5-turbo",
    )

    return bot_response, state
