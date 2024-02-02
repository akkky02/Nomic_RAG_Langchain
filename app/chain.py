import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate






urls = [
    "https://weber-stephen.medium.com/unleashing-the-ultimate-ai-battle-llama-2-vs-chatgpt-gpt-3-5-a-creative-showdown-9919608200d7",
    "https://medium.com/@datadrifters/llama-2-unleashed-metas-new-open-source-commercially-free-answer-to-advanced-ai-b1c7a0f32865",
    "https://medium.com/the-generator/battle-of-the-bots-chatgpt-vs-claude-2-vs-llama-2-2728083b6008",
    "https://ai.gopubby.com/mixtral-8x7b-vs-llama2-70b-c9626a5aec4d",
    "https://towardsdatascience.com/mistral-ai-vs-meta-comparing-top-open-source-llms-565c1bc1516e",
    "https://medium.com/mlearning-ai/the-ultimate-showdown-mistral7b-vs-llama2-13b-unveiling-the-champion-cdbde67abbde",
    "https://medium.com/illuminations-mirror/zephyr-7b-%CE%B1-vs-llama-2-70b-vs-mistral-7b-unraveling-the-future-of-ai-language-models-a34d95968f40",
    "https://medium.com/@GenerationAI/codellama-70b-vs-34b-vs-mistral-medium-vs-gpt-4-ec7d1739af6a",
    "https://medium.com/educative/google-gemini-vs-chatgpt-everything-we-know-so-far-7af259fe3022",
    "https://medium.com/@shariq.ahmed525/i-tested-gemini-and-chatgpt4-to-find-which-one-is-best-61d3edc8deff"
    ]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=7500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1")
)
retriever = vectorstore.as_retriever()

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM API
# model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# Local LLM
ollama_llm = "mistral:instruct"
model_local = ChatOllama(model=ollama_llm)

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)