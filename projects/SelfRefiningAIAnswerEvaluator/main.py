from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()


class SelfRefining:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.parser = StrOutputParser()

    def generate_initial_answer(self, question: str) -> str:
        prompt = PromptTemplate(
            template="Give answer to user query {input}",
            input_variables=["input"],
        )
        chain = prompt | self.model | self.parser
        return chain.invoke({"input": question})

    def reviewer(self, answer: str) -> str:
        quality_prompt = PromptTemplate(
            template="Evaluate the quality of the initial answer {input}",
            input_variables=["input"],
        )
        clarity_prompt = PromptTemplate(
            template="Evaluate the clarity of the initial answer {input}",
            input_variables=["input"],
        )
        completeness_prompt = PromptTemplate(
            template="Evaluate the completeness of the initial answer {input}",
            input_variables=["input"],
        )

        parallel_chain = RunnableParallel(
            {
                "quality": quality_prompt | self.model | self.parser,
                "clarity": clarity_prompt | self.model | self.parser,
                "completeness": completeness_prompt | self.model | self.parser,
            }
        )

        evaluations = parallel_chain.invoke({"input": answer})

        combine_prompt = PromptTemplate(
            template=(
                "Combine the following evaluation results into a single, "
                "actionable feedback (150–200 words): {result}"
            ),
            input_variables=["result"],
        )

        combine_chain = combine_prompt | self.model | self.parser
        return combine_chain.invoke({"result": evaluations})

    def improve_answer(self, answer: str, feedback: str) -> str:
        improve_prompt = PromptTemplate(
            template=(
                "Improve the following answer based on the given feedback.\n\n"
                "Answer:\n{input}\n\n"
                "Feedback:\n{feedback}"
            ),
            input_variables=["input", "feedback"],
        )

        improve_chain = improve_prompt | self.model | self.parser
        return improve_chain.invoke({"input": answer, "feedback": feedback})

    def run(self, question: str):
        initial_answer = self.generate_initial_answer(question)
        feedback = self.reviewer(initial_answer)
        improved_answer = self.improve_answer(initial_answer, feedback)

        return {
            "initial_answer": initial_answer,
            "feedback": feedback,
            "improved_answer": improved_answer,
        }


if __name__ == "__main__":
    question = input("Enter your question: ")

    refiner = SelfRefining()
    result = refiner.run(question)

    print("\nInitial Answer:\n", result["initial_answer"])
    print("\nFeedback:\n", result["feedback"])
    print("\nImproved Answer:\n", result["improved_answer"])
