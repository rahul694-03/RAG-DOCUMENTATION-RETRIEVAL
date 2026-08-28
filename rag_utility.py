
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


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Check Groq API key
if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please create a .env file and add your Groq API key."
    )


# =========================================================
# Working Directory
# =========================================================

working_dir = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# HuggingFace Embedding Model
# =========================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# Groq Llama 3.3 70B
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY

)


# =========================================================
# Chroma Database Path
# =========================================================

VECTORSTORE_PATH = os.path.join(
    working_dir,
    "doc_vectorstore"
)


# =========================================================
# Process PDF
# =========================================================

def process_document_to_chroma_db(file_name):

    pdf_path = os.path.join(
        working_dir,
        file_name
    )

    # -----------------------------------------------------
    # Load PDF
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
    # Split Text
    # -----------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
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
    # Create Chroma Vector Database
    # -----------------------------------------------------

    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_PATH
    )

    return True


# =========================================================
# Answer Question
# =========================================================

def answer_question(user_question):

    # -----------------------------------------------------
    # Load Chroma Database
    # -----------------------------------------------------

    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding
    )


    # -----------------------------------------------------
    # Create Retriever
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

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context, say:

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
    # Format Retrieved Documents
    # -----------------------------------------------------

    def format_docs(docs):

        return "\n\n".join(
            document.page_content
            for document in docs
        )


    # -----------------------------------------------------
    # RAG Chain
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
    # Generate Answer
    # -----------------------------------------------------

    answer = rag_chain.invoke(
        user_question
    )

    return answer

