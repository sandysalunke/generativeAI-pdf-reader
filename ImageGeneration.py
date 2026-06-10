import streamlit as st
from dotenv import load_dotenv
import requests
from helpers.llm import generate_image
load_dotenv()

# Display chat
def display_chat():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            if role == "assistant" and isinstance(msg, bytes):
                st.write("Here is your generated image:")
                st.image(msg)
            else:
                st.write(msg)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Image Generator", layout="centered")

st.title("🎨 AI Image Generator (Azure OpenAI)")

style = st.selectbox(
    "Select Style",
    ["Realistic", "Cartoon", "3D", "Anime", "Cyberpunk"]
)

prompt = st.chat_input("A futuristic Mumbai skyline at sunset")
display_chat()
if prompt:
    
    st.session_state.chat_history.append(("user", prompt))
    st.chat_message("user").write(prompt)
    
    # Placeholder for spinner (important)
    spinner_placeholder = st.empty()

    # Show spinner ABOVE input using placeholder
    with spinner_placeholder:
        with st.spinner("Generating image..."):
            final_prompt = f"{style} style: {prompt}"
            image_bytes = generate_image(final_prompt)

    # Clear spinner (optional)
    spinner_placeholder.empty()

    #Add Image to chat history
    st.session_state.chat_history.append(("assistant", image_bytes))

    st.rerun()
    