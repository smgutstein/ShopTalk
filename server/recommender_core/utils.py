import logging
import os

from dotenv import load_dotenv

PERSONALITIES = [
    "Mae West",
    "Pentecostal preacher",
    "1920s flapper",
    "1950s beatnik",
    "1960s hippie",
    "Puritan preacher",
    "Damon Runyon character",
    "Shakespearean character",
    "Dickensian character",
    "1920s gangster",
    "1950s greaser",
    "Edward Bulwer-Lytton character",
]


def elapsed_time_string(start_time, stop_time):
    delta_time = stop_time - start_time
    minutes, seconds = divmod(delta_time.seconds, 60)
    return minutes, seconds, f"{minutes} minutes, {seconds} seconds"


def load_openai_api_key():
    """Load and validate the OpenAI API key before expensive setup work."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    return api_key


def serialize_convo(conversation_history):
    return [
        {"type": msg.__class__.__name__, "content": msg.content}
        for msg in conversation_history
    ]

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
