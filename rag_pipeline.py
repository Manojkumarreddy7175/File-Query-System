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


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_answer(vectorstore, query: str, api_key: str, model: str) -> dict:
    """
    Retrieve relevant chunks from FAISS and generate an answer with the Groq LLM.
    Returns a dict with 'result' (answer string) and 'source_documents'.
    """
    llm = ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=0.2,
        max_tokens=1024,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Retrieve source docs first so we can return them
    source_docs = retriever.invoke(query)

    chain = (
        {"context": lambda _: _format_docs(source_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)

    return {"result": answer, "source_documents": source_docs}
