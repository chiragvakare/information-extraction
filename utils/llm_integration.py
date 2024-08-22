# utils/llm_integration.py
import os
import google.generativeai as generativeai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Generative AI API with your API key
generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini model
model = generativeai.GenerativeModel("gemini-pro")

def start_chat():
    return model.start_chat(history=[])

def get_gemini_response(chat, prompt):
    response = chat.send_message(prompt, stream=True)
    response_text = ""
    for chunk in response:
        response_text += chunk.text + "\n"
    return response_text.strip()

def query_llm(context, query):
    try:
        # Start a chat session
        chat = start_chat()
        
        # Combine context with the query to form a comprehensive prompt
        prompt = f"Context: {context}\n\nQuestion: {query}"
        
        # Get the response from the Gemini model
        response_text = get_gemini_response(chat, prompt)
        
        # Return the response
        return response_text

    except Exception as e:
        return f"An error occurred: {e}"
