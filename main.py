import os
import sys
from tabnanny import verbose
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-weary-lever-12"
import getpass
import os
os.environ["LANGCHAIN_API_KEY"]="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"
from langchain_ollama import ChatOllama  # Updated import
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import pandas as pd
from langchain.docstore.document import Document
import re
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_experimental.agents import create_pandas_dataframe_agent
import tabulate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader

# Load your CSV file
df = CSVLoader(
    r"C:\Users\yyuej\Desktop\AI实习\RAG2\movie.csv",
    encoding="utf-8"
)
# Generalize document creation based on DataFrame columns
documents = df.load()
#print (documents)
# Initialize the language model
llm = ChatOllama(model="llama3.2", callbacks=[StreamingStdOutCallbackHandler()],temperture=0.1)

# Initialize embeddings and vector store
embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()

# Define the QA prompt
template = """
You are an AI assistant that provides helpful answers based on the provided context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
)


from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "document search",
    "You are an assistant for finding information tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
)


# Define the tools for the agent
tools = [
    Tool(
        name="QA Tool",
        func=retriever_tool,
        description="Search for documents based on a query. All the questions is base on the data in the document.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.If the document does not have that data, just say that the document is not found.",
    ),
]
myagent=initialize_agent(
    llm=llm,
    tools =tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    return_intermediate_steps=False,
    handle_parsing_errors=True,)
# agent_executor = AgentExecutor(agent=myagent, tools=tools, verbose=True, handle_parsing_errors=True,return_intermediate_steps=False,)

# Start the query loop
while True:
    query = input("Query: \n")
    if query.lower() == "exit":
        break
    if query.strip() == "":
        continue
    myagent.invoke({"input": query})

    print("\n")