# %% [markdown]
# # **Contextual RAG**
#
# Contextual Retrieval-Augmented Generation (RAG) is an advanced RAG technique that improves
# response relevance and efficiency by incorporating contextual compression during the retrieval
# process. Traditional RAG retrieves and sends full documents to the generation model, which may
# include irrelevant information, leading to higher costs and less accurate responses.
#
# In Contextual RAG, the retrieved documents are processed through a Document Compressor before
# being passed to the language model. This compressor extracts and retains only the most relevant
# information for the query, or even discards entire irrelevant documents.
#
# Reference: https://python.langchain.com/docs/how_to/contextual_compression/

# %% [markdown]
# ## **Initial Setup**

# %%
import os
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
import chromadb
from langchain_chroma import Chroma

chroma_client = chromadb.EphemeralClient()
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    client=chroma_client,
    collection_name="contextual_rag",
)

# %% [markdown]
# ## **Retriever**

# %%
# create retriever
retriever = vectorstore.as_retriever()

# %% [markdown]
# ## **Contextual Retriever**

# %%
# create llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# %%
# create compression retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# %%
# checking compressed docs
compressed_docs = compression_retriever.invoke("what are points on a mortgage")
compressed_docs

# %% [markdown]
# ## **RAG Chain**

# %%
# create rag chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": compression_retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
# response
response = rag_chain.invoke("what are points on a mortgage")
response

# %% [markdown]
# ## **Preparing Data for Evaluation**

# %%
# run evaluation queries and collect results
questions = ["what are points on a mortgage"]

data = {"query": [], "response": [], "context": []}

for query in questions:
    resp = rag_chain.invoke(query)
    context = [doc.page_content for doc in compression_retriever.invoke(query)]
    data["query"].append(query)
    data["response"].append(resp)
    data["context"].append(context)

import pandas as pd
df = pd.DataFrame(data)
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
        "user_input": q,
        "response": r,
        "retrieved_contexts": c,
    }
    for q, r, c in zip(data["query"], data["response"], data["context"])
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
