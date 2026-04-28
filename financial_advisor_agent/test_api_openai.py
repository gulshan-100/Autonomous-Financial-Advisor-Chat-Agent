from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
import os 

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model = 'gpt-4o',
    temperature=0.4,
    api_key=api_key
)

response = llm.invoke("Hello")

print(response.content)