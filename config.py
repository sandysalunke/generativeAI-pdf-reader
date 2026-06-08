import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# --- Azure Config ---
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
DATA_PATH = "data/"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_ENDPOINT
)