from config import client, CHAT_MODEL, IMAGE_MODEL
import base64

def ask_llm(query, context):
    prompt = f"""
    Answer only using the context.

    Context:
    {context}

    Question:
    {query}
    """

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an enterprise assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

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
                4. List of attendees (if mentioned)

                Transcript:
                {transcript}
                """
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_image(prompt):
    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024"
    )
    
    return base64.b64decode(result.data[0].b64_json)