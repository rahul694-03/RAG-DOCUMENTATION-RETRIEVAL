import os
import streamlit as st

from rag_utility import (
    process_document_to_chroma_db,
    answer_question
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF QA Bot",
    page_icon="🦙",
    layout="wide"
)


# ---------------------------------------------------------
# Working directory
# ---------------------------------------------------------
working_dir = os.path.dirname(
    os.path.abspath(__file__)
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🦙 Llama - Document RAG")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    save_path = os.path.join(
        working_dir,
        uploaded_file.name
    )

    # Save PDF
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process PDF
    try:

        with st.spinner(
            "Processing PDF... This may take a moment."
        ):

            process_document_to_chroma_db(
                uploaded_file.name
            )

        st.success(
            "✅ PDF processed successfully!"
        )

    except Exception as e:

        st.error(
            f"❌ Error processing PDF: {e}"
        )


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------
st.subheader("Ask a question")

user_question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is this document about?"
)


# ---------------------------------------------------------
# Answer
# ---------------------------------------------------------
if st.button("🔍 Answer"):

    if not user_question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not os.path.exists(
        os.path.join(
            working_dir,
            "doc_vectorstore"
        )
    ):

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        try:

            with st.spinner(
                "🦙 Generating answer..."
            ):

                answer = answer_question(
                    user_question
                )

            st.markdown(
                "### 🦙 Answer"
            )

            st.write(answer)

        except Exception as e:

            st.error(
                f"❌ Error generating answer: {e}"
            )