import argparse
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

#database logic
PERSIS_PATH = "./qdrant_db"
COLLECTION_NAME = "pirates-of-the-caribbean-scripts"

def load_script(url, movie_title):
    response = requests.get(url) #uses request package to get the url

    #Scrapes the script from the page
    soup = BeautifulSoup(response.content, "html.parser") #
    script_raw = soup.find("pre").get_text()

    #create a langchain document based on it
    return Document(page_content=script_raw, metadata={"title": movie_title}) #the more metadata i.e. character name etc. provided the more intelligent the agent becomes

def ingest_scripts():
    pirates_of_the_caribbean_scripts = [
        {
            "title": "Pirates of the Caribbean: The Curse of Black Pearl",
            "url": "https://imsdb.com/scripts/Pirates-of-the-Caribbean.html",
        },
        {
            "title": "Pirates of the Caribbean: Dead Man's Chest",
            "url": "https://imsdb.com/scripts/Pirates-of-the-Caribbean-Dead-Man's-Chest.html",
        },
        {
            "title": "Pirates of the Caribbean: At World's End",
            "url": "http://www.script-o-rama.com/movie_scripts/a2/pirates--of-the-caribbean-worlds-end-script.html",
        },
    ]
    script_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        add_start_index=True,
        separators=["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""],
    )
    all_chunks = []
    for script in pirates_of_the_caribbean_scripts:
        doc = load_script(script["url"], script["title"])
        chunks = script_splitter.split_documents([doc])
        all_chunks.extend(chunks)
        print(f"Loaded and split script for {script['title']} into {len(chunks)} chunks.")

    QdrantVectorStore.from_documents(
        all_chunks,
        embedding=embeddings,
        path=PERSIS_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"Embedded {len(all_chunks)} chunks with {EMBEDDING_MODEL} into {PERSIS_PATH}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Replace the local Qdrant collection and embed all scripts again.",
    )
    args = parser.parse_args()
    client = QdrantClient(path=PERSIS_PATH)

    if args.reembed:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.close()
        ingest_scripts()
        return

    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        ingest_scripts()
        client = QdrantClient(path=PERSIS_PATH)
    vectorstore = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        client=client,
    )
    # Both embeddings and chat run locally; Hugging Face downloads are free and
    # do not require an API key. The first run downloads the chat model weights.
    local_llm = HuggingFacePipeline.from_model_id(
        model_id=CHAT_MODEL,
        task="text-generation",
        device=-1,
        pipeline_kwargs={"max_new_tokens": 512, "do_sample": False},
    )
    line = ChatHuggingFace(llm=local_llm)
    #fetches the top 15 most relevant chunks from the vectorstore based on the query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    template = """
    You are a Pirates of the Caribbean Movie Scipt Expert. Use ONLY the following excerpts to answer.
    If the answer isn't in the context, say "There is no Information about this in the original Pirates of the Caribbean scripts."

    Context:
    {context}

    Question: 
    {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    #chains together multiple llm calls
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | line
        | StrOutputParser()
    )

    print("\n--- The Pirates of the Caribbean Expert is ready to answer your questions --- \n")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nPirates of the Caribbean Expert: {response}")

if __name__ == "__main__":
    main()
