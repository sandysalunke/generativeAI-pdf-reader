'''
FLOW:
Audio File
   ↓
Speech-to-Text (Whisper)
   ↓
Transcript
   ↓
LLM Processing
   ├── Summary
   ├── Key Points
   ├── Action Items
   ↓
UI Output (Streamlit)

TECH:
Python
OpenAI / Whisper
LangChain (optional)
Streamlit (UI)
'''

from openai import AzureOpenAI
import streamlit as st
import os
from dotenv import load_dotenv

# --- Azure Config ---
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_ENDPOINT
)

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL   # or your Azure deployment name
        )
    return transcript.text

def process_meeting(transcript):
    response = client.chat.completions.create(
        model=CHAT_MODEL,  # your deployment name
        messages=[
            {
                "role": "system",
                "content": "You are an AI meeting assistant."
            },
            {
                "role": "user",
                "content": f"""
                Analyze this meeting transcript:

                Provide:
                1. Summary
                2. Key Points
                3. Action Items (with owner if possible)

                Transcript:
                {transcript}
                """
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("AI Meeting Assistant")

uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav"])

if uploaded_file:
    with open("audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Transcribing...")
    transcript = transcribe_audio("audio.wav")

    st.subheader("Transcript")
    st.write(transcript)

    st.info("Analyzing...")
    insights = process_meeting(transcript)

    st.subheader("Meeting Insights")
    st.write(insights)
