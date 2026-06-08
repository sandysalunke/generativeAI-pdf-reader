from config import client, WHISPER_MODEL

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL   # or your Azure deployment name
        )
    return transcript.text
