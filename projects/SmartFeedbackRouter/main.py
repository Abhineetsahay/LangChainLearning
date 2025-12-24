from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

sentiment_prompt = PromptTemplate(
    template="""
    Classify the sentiment of the following text as
    Positive, Neutral, or Negative.
    
    Text: {text}
    """,
    input_variables=["text"],
)

sentiment_chain = sentiment_prompt | model | parser

positive_prompt = PromptTemplate(
    template="The user is happy. Respond politely and thank them.\nText: {text}",
    input_variables=["text"],
)

neutral_prompt = PromptTemplate(
    template="The user is neutral. Provide a clear and helpful response.\nText: {text}",
    input_variables=["text"],
)

negative_prompt = PromptTemplate(
    template="The user is unhappy. Provide a solution.\nText: {text}",
    input_variables=["text"],
)

positive_chain = positive_prompt | model | parser
neutral_chain = neutral_prompt | model | parser
negative_chain = negative_prompt | model | parser

router = RunnableBranch(
    (lambda x: "Positive" in x["sentiment"], positive_chain),
    (lambda x: "Negative" in x["sentiment"], negative_chain),
    neutral_chain,  
)

pipeline = (
    RunnableLambda(
        lambda x: {
            "text": x,
            "sentiment": sentiment_chain.invoke({"text": x}),
        }
    )
    | router
)

user_input = input("Enter your query: ")
result = pipeline.invoke(user_input)

print(result)
