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
from langchain_experimental.agents import create_pandas_dataframe_agent
import tabulate


df = pd.read_csv(
    r"C:\Users\yyuej\Desktop\AI实习\agency\agent\movie.csv",
    delimiter=",",
    quotechar='"',
    names=["title", "rating", "plot"],
    encoding="utf-8"
)

documents = []
for _, row in df.iterrows():
    content = f"Title: {row['title']}\nRating: {row['rating']}\nPlot: {row['plot']}"
    documents.append(Document(page_content=content))

llm = ChatOllama(model="mistral", callbacks=[StreamingStdOutCallbackHandler()])

embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

agent = create_pandas_dataframe_agent( llm,
                                       df,
                                       verbose=True,
                                       agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                       allow_dangerous_code=True, )


while True:
    query = input("Query: \n")
    if query.lower() == "exit":
        break
    if query.strip() == "":
        continue

    try:
        if re.search(r"\b(平均|最大|最小|统计|绘图|数据分析)\b", query):
            result = agent.invoke(query)
        else:
            result = qa_chain.invoke(query)
    except Exception as e:
        print("An error occurred:", e)
    print("\n")