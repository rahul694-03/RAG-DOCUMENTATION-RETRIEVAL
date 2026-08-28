import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    UnstructuredPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.runnables import (
    RunnablePassthrough
)

from langchain_core.output_parsers import (
    StrOutputParser
)


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

working_dir = os.path.dirname(
    os.path.abspath(__file__)
)


# ---------------------------------------------------------
# Check Groq API key
# ---------------------------------------------------------
groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

if not groq_api_key:

    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please add it to your .env file."
    )


# ---------------------------------------------------------
# HuggingFace Embeddings
# ---------------------------------------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0

)


# ---------------------------------------------------------
# Chroma database path
# ---------------------------------------------------------
VECTORSTORE_PATH = os.path.join(
    working_dir,
    "doc_vectorstore"
)


# ---------------------------------------------------------
# Process PDF
# ---------------------------------------------------------
def process_document_to_chroma_db(file_name):

    pdf_path = os.path.join(
        working_dir,
        file_name
    )

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )


    # -----------------------------------------------------
    # Load PDF using UnstructuredPDFLoader
    # -----------------------------------------------------
    loader = UnstructuredPDFLoader(
        pdf_path
    )

    documents = loader.load()


    if not documents:

        raise ValueError(
            "Could not extract text from the PDF."
        )


    # -----------------------------------------------------
    # Split text
    # -----------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(
        documents
    )


    if not texts:

        raise ValueError(
            "No text chunks were created from the PDF."
        )


    # -----------------------------------------------------
    # Create Chroma vector database
    # -----------------------------------------------------
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_PATH
    )


    return True


# ---------------------------------------------------------
# Answer question
# ---------------------------------------------------------
def answer_question(user_question):

    if not os.path.exists(
        VECTORSTORE_PATH
    ):

        raise ValueError(
            "Vector database does not exist. "
            "Please upload a PDF first."
        )


    # -----------------------------------------------------
    # Load Chroma
    # -----------------------------------------------------
    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding
    )


    # -----------------------------------------------------
    # Retriever
    # -----------------------------------------------------
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information
contained in the context below.

If the answer is not available in the context,
say exactly:

"I could not find the answer in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
    )


    # -----------------------------------------------------
    # Format documents
    # -----------------------------------------------------
    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    # -----------------------------------------------------
    # RAG chain
    # -----------------------------------------------------
    rag_chain = (

        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }

        | prompt

        | llm

        | StrOutputParser()
    )


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------
    answer = rag_chain.invoke(
        user_question
    )


    return answer