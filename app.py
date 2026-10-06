
import os
import hashlib
from io import BytesIO

import streamlit as st
import chromadb
import numpy as np 

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai 
from dotenv import load_dotenv


# ---------------- CONFIGURATION ----------------

load_dotenv()

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Assistant")
st.write("Upload your study PDF and ask questions using AI.")

API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL_NAMES = [
    model.strip()
    for model in os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash,gemini-3.8-flash"
    ).split(",")
    if model.strip()
]


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


embedding_model = load_embedding_model()
chroma_client = load_chroma_client()


# ---------------- TEXT PROCESSING ----------------

def split_text(text, chunk_size=700, overlap=100):
    """Split text into smaller overlapping chunks."""
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def process_pdf(pdf_bytes, collection):
    """Extract PDF text, create embeddings and store in ChromaDB."""
    reader = PdfReader(BytesIO(pdf_bytes))

    documents = []
    metadatas = []
    ids = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(split_text(page_text)):
            documents.append(chunk)
            metadatas.append({
                "page": page_number
            })
            ids.append(f"page_{page_number}_chunk_{chunk_number}")

    if not documents:
        return 0

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


# ---------------- RETRIEVAL ----------------

def retrieve_context(question, collection, top_k=3):
    """Find the most relevant chunks for the question."""
    total_documents = collection.count()
    if total_documents == 0:
        return "", []

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_documents),
        include=["documents", "metadatas", "distances"]
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    context_parts = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        page = metadata.get("page", "Unknown")
        context_parts.append(f"[Page {page}]\n{document}")
        sources.append(page)

    return "\n\n".join(context_parts), sorted(set(sources))


# ---------------- AI ANSWER ----------------

def generate_answer(question, context):
    """Generate an answer using Gemini and retrieved context."""
    if not API_KEY:
        raise ValueError(
            "Gemini API key not found. Add GEMINI_API_KEY to your .env file."
        )

    client = genai.Client(api_key=API_KEY)

    prompt = f"""
You are a helpful AI Study Assistant.

Answer the student's question using only the context
provided below.

If the answer is not available in the context, clearly
say that the uploaded document does not contain enough
information to answer the question.

Explain the answer in simple, student-friendly language.
Do not make up facts.

CONTEXT:
{context}

STUDENT'S QUESTION:
{question}

ANSWER:
"""

    last_error = None
    for model_name in DEFAULT_MODEL_NAMES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text or "No answer was generated."
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "not_found" not in message and "unavailable" not in message:
                raise

    if last_error is not None:
        raise RuntimeError(
            "Gemini API is temporarily unavailable. Please try again in a moment."
        ) from last_error

    raise RuntimeError("No Gemini model was available for generation.")


# ---------------- STREAMLIT INTERFACE ----------------

uploaded_file = st.file_uploader(
    "Upload your study PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = f"study_{file_hash}"

    st.info(f"Selected file: {uploaded_file.name}")

    if st.button("📄 Process PDF", type="primary"):
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                collection = chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                if collection.count() == 0:
                    chunk_count = process_pdf(pdf_bytes, collection)
                else:
                    chunk_count = collection.count()

                if chunk_count == 0:
                    st.error(
                        "No readable text found. If this is a scanned PDF, "
                        "you may need OCR."
                    )
                    st.session_state.pop("collection_name", None)
                else:
                    st.session_state["collection_name"] = collection_name
                    st.session_state["file_name"] = uploaded_file.name
                    st.success(
                        f"PDF processed successfully! "
                        f"{chunk_count} text chunks are ready."
                    )

        except Exception as e:
            st.error(f"Error processing PDF: {e}")


# ---------------- QUESTION AND ANSWER ----------------

if "collection_name" in st.session_state:
    st.divider()
    st.subheader("💬 Ask your question")

    st.caption(f"Study material: {st.session_state['file_name']}")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: Explain machine learning in simple words."
    )

    if st.button("🤖 Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                collection = chroma_client.get_collection(
                    name=st.session_state["collection_name"]
                )

                with st.spinner("Searching your PDF and generating answer..."):
                    context, sources = retrieve_context(
                        question,
                        collection
                    )

                    if not context:
                        st.warning(
                            "No readable text was found in the uploaded PDF. "
                            "Please process a different PDF or ensure the document is text-based."
                        )
                    else:
                        answer = generate_answer(question, context)

                if context:
                    st.markdown("### 📝 Answer")
                    st.write(answer)

                st.markdown("**📖 Source pages:**")
                if sources:
                    st.write(", ".join(map(str, sources)))
                else:
                    st.write("No source pages available.")

                with st.expander("View retrieved text"):
                    st.write(context)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption("AI Study Assistant | Built with Python, RAG, ChromaDB and Gemini")