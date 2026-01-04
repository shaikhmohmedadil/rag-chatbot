from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Testing OpenAI API connection...")
print("This will cost about $0.0001 (basically free!)")

try:
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello! API is working!' in one sentence."}
        ],
        max_tokens=50
    )

    print("Great")
    print()
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Error Message: {e}")    