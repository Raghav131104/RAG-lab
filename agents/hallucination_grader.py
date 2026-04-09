from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Grader 1: Hallucination Grader
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)

system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination


# Grader 2: Answer Grader
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

system_answer = """You are a grader assessing whether an answer addresses / resolves a question. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
     If the question has multiple parts and the answer resolves at least ONE of the parts, score it 'yes'."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_grader_answer
