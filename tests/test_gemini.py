from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

try:
    # Configure genai
    genai.configure(api_key=api_key)
    
    # Try to list models
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Test message')
    print("API Test successful!")
    print("Response:", response.text)
    
except Exception as e:
    print(f"Error: {str(e)}")