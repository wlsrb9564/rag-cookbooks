# %% [markdown]
# # **Adaptive RAG**
#
# Adaptive RAG dynamically selects the retrieval strategy based on the query type.
# A Question Router decides whether to use the internal vectorstore or web search.
# Retrieved documents are graded for relevance, and the final generation is checked
# for hallucinations and answer quality using dedicated grader chains.
#
# Flow: Route → Retrieve/WebSearch → Grade Docs → Generate → Grade Generation → END

# %% [markdown]
# ## **Initial Setup**

# %%
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# ## **Indexing**

# %%
# load embedding model
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %%
# load data
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("../data/context.csv", encoding="utf-8")
documents = loader.load()

# %%
# split documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# %%
# create vectorstore
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents, embeddings)

# %% [markdown]
# ## **Retriever**

# %%
# create retriever
retriever = vectorstore.as_retriever()

# %% [markdown]
# ## **Question Router**
# Routes a query to either the internal vectorstore or web search.

# %%
# create question router
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to either a vectorstore or web search.
The vectorstore contains information on the following topics:
- Finance and real estate
- Library and research topics
- Biology and microbiology
- Literature and writing
- Movies and entertainment
- Animals and nature
- History and geography
- Astronomy

If the question is related to these topics, route it to the vectorstore. Otherwise, use web search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# %%
question_router.invoke({"question": "how does interlibrary loan work?"})

# %%
question_router.invoke({"question": "What is RAG?"})

# %% [markdown]
# ## **Document Grader**
# Evaluates whether a retrieved document is relevant to the query.

# %%
# create document grader
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# %%
# testing grader example 1
question = "how does interlibrary loan work"
docs = retriever.invoke(question)
print(retrieval_grader.invoke({"question": question, "document": docs}))

# %%
# testing grader example 2
# question = "What is RAG?"
# docs = retriever.invoke(question)
# print(retrieval_grader.invoke({"question": question, "document": docs}))

# %% [markdown]
# ## **RAG Chain**

# %%
# create RAG chain
from langchain_core.output_parsers import StrOutputParser

template = """You are a helpful assistant that answers questions based on the following context.
Use the provided context to answer the question.
Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

# %%
# response
generation = rag_chain.invoke({"context": docs, "question": question})
generation

# %% [markdown]
# ## **Hallucination Grader**
# Checks whether the answer is grounded in the retrieved documents.

# %%
# create hallucination grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})

# %% [markdown]
# ## **Answer Grader**
# Evaluates whether the answer effectively addresses the question.

# %%
# create answer grader
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

# %% [markdown]
# ## **Web Search**

# %%
# define web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

# %% [markdown]
# ## **Create Graph**
# ### **Define Graph State**

# %%
# define graph state
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# %% [markdown]
# ### **Define Nodes and Edges**

# %%
# define nodes and edges
from langchain_core.documents import Document

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}


def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# %% [markdown]
# ### **Build Graph**

# %%
# build graph
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": END,  # simplified: end if no relevant docs found
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": END,
    },
)

app = workflow.compile()

# %%
# example 1: vectorstore route (relevant documents)
inputs = {"question": "how does interlibrary loan work"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])

# %%
# example 2: web search route (non-relevant documents)
inputs = {"question": "What is RAG?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])

# %% [markdown]
# ## **Preparing Data for Evaluation**

# %%
# run graph and collect results for evaluation
inputs = {"question": "how does interlibrary loan work"}
outputs = []

for output in app.stream(inputs):
    for key, value in output.items():
        if key == "generate":
            question = value["question"]
            documents = value["documents"]
            generation = value["generation"]
            outputs.append({
                "query": question,
                "context": [doc.page_content for doc in documents],
                "response": generation,
            })

import pandas as pd
df = pd.DataFrame(outputs)
df

# %% [markdown]
# ## **Evaluation with Ragas**
#
# [Ragas](https://docs.ragas.io/)로 RAG 파이프라인을 평가합니다.
#
# - **Faithfulness**: 답변이 검색된 컨텍스트에 근거하는지 (환각 탐지)
# - **Answer Relevancy**: 답변이 질문에 얼마나 관련 있는지

# %%
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy

eval_data = [
    {
        "user_input": row["query"],
        "response": row["response"],
        "retrieved_contexts": row["context"],
    }
    for row in outputs
]
ragas_dataset = EvaluationDataset.from_dict(eval_data)

# %%
result = evaluate(
    dataset=ragas_dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
)
print(result)

# %%
result.to_pandas()
