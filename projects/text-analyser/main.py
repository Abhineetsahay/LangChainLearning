from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

parser = StrOutputParser()

sentiment_prompt = PromptTemplate(
    template="""
    Classify the sentiment of the following text as
    Positive, Neutral, or Negative.
    
    Text: {text}
    """,
    input_variables=["text"],
)

summary_prompt = PromptTemplate(
    template="Provide a brief summary of the following text:\n{text}",
    input_variables=["text"],
)

facts_prompt = PromptTemplate(
    template="Generate 5 important points from the following text:\n{text}",
    input_variables=["text"],
)

parallel_chain = RunnableParallel(
    {
        "sentiment": sentiment_prompt | model | parser,
        "brief_idea": summary_prompt | model | parser,
        "important_points": facts_prompt | model | parser,
    }
)

result = parallel_chain.invoke({"text": "I am very happy as I got placed"})

print(result)
