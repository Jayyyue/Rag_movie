import os
import sys
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

# Load your CSV file
df = pd.read_csv(
    r"movie.csv",
    delimiter=",",
    quotechar='"',
    encoding="utf-8"
)

# Generalize document creation based on DataFrame columns
documents = []
for _, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=content))

# Initialize the language model
llm = ChatOllama(model="mistral", callbacks=[StreamingStdOutCallbackHandler()])

# Initialize embeddings and vector store
embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Define the QA prompt
template = """You are an AI assistant that provides helpful answers based on the provided context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# Create the DataFrame agent
dataframe_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    return_intermediate_steps=False,
    allow_dangerous_code=True
)

# Define the tools for the agent
tools = [
    Tool(
        name="DataFrame Tool",
        func=dataframe_agent.run,
        description=(
            "Useful for when you need to perform data analysis, computation, or generate plots "
            "based on the CSV data."
        ),
    ),
    Tool(
        name="RetrievalQA Tool",
        func=qa_chain.run,
        description=(
            "Useful for when you need to answer questions or provide information based on the content "
            "of the documents."
        ),
    ),
]

# Initialize the agent with the tools
agent = create_pandas_dataframe_agent( llm,
                                       df,
                                       verbose=False,
                                       agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                       max_iterations=3,
                                       allow_dangerous_code=True,
                                       agent_executor_kwargs={"handle_parsing_errors": True},
                                       return_intermediate_steps=False,
                                       )

# Start the query loop
while True:
    query = input("Query: \n")
    if query.lower() == "exit":
        break
    if query.strip() == "":
        continue

    try:
        message = AIMessage(content='\`\`\`\n{"foo": "bar"}\n\`\`\`')
        output_parser = JsonOutputParser()
        result = agent.invoke(query)
        print("\n")
    except Exception as e:
        print(f"An error occurred: {e}")