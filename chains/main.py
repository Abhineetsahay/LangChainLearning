from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt=PromptTemplate(
    template='Generate 5 interesting facts about {topic} and each fact should be in 1-2 lines',
    input_variables=['topic']
)


model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

parser=StrOutputParser()

chain=prompt | model | parser

result = chain.invoke({"topic":"india"})

print(result)

chain.get_graph().print_ascii()