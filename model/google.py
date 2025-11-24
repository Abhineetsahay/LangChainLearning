from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


prompt='create a poem about love in 1-2 lines'

result = model.invoke(prompt)

print("default model ",result.content)


model_creative=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # api_key=api_key,
    temperature=0.5
)

creative_result=model.invoke(prompt)

print("\ncreative model", creative_result.content)
