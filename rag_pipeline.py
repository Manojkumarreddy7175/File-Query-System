import os
import tempfile
import pypdf
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_text(file_path: str, file_type: str) -> str:
    """Extract plain text from a PDF, DOCX, or TXT file."""
    file_type = file_type.lower().lstrip(".")

    if file_type == "pdf":
        reader = pypdf.PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    elif file_type == "docx":
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported file type: {file_type}")


# ── Vector Store ─────────────────────────────────────────────────────────────

def build_vectorstore(text: str):
    """
    Split text into chunks, embed them with a local HuggingFace model,
    and store in an in-memory FAISS index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── Answer Generation ─────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a knowledgeable assistant. Using only the context below, \
answer the question in a clear, well-structured way.

If the answer cannot be found in the context, respond with:
"I couldn't find relevant information about that in the document."

Context:
{context}

Question: {question}

Answer:"""


# Cosine distance threshold: 0 = identical, 2 = opposite.
# Chunks with distance above this value are considered irrelevant.
SIMILARITY_THRESHOLD = 0.75
OUT_OF_SCOPE_MSG = (
    "I couldn't find relevant information about that in the document. "
    "Please ask something related to the uploaded file."
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_answer(vectorstore, query: str, api_key: str, model: str) -> dict:
    """
    Retrieve relevant chunks from FAISS and generate an answer with the Groq LLM.
    Returns a dict with 'result' (answer string) and 'source_documents'.

    If the closest chunks are semantically too far from the query
    (distance > SIMILARITY_THRESHOLD), returns an out-of-scope message
    without calling the LLM at all.
    """
    # ── Score-aware retrieval ─────────────────────────────────────────────────
    # Returns list of (Document, float) where float is L2 distance.
    # Lower distance = more similar. With normalised embeddings this equals
    # (2 - 2*cosine_similarity), so distance < 0.75 ≈ cosine_sim > 0.625.
    scored_docs = vectorstore.similarity_search_with_score(query, k=4)

    # Filter to only chunks that are actually relevant
    relevant_docs = [
        doc for doc, score in scored_docs if score < SIMILARITY_THRESHOLD
    ]

    if not relevant_docs:
        return {"result": OUT_OF_SCOPE_MSG, "source_documents": []}

    # ── LLM answer generation ─────────────────────────────────────────────────
    llm = ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": lambda _: _format_docs(relevant_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    return {"result": answer, "source_documents": relevant_docs}
