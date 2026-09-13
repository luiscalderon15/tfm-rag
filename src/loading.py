from pathlib import Path
import re
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
from langchain_core.documents import Document
import logging
import time
import json
from config import DATA_FOLDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(asctime)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("docling").setLevel(logging.WARNING)
def count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    else:
        words = re.findall(r'\b\w+\b', text)
        return len(words)

# def load_pdf(file_path:Path)->Document: # Load data loader = PyPDFLoader(file_path) # List of documents docs = loader.load() # Total pages CV n_pages = len(docs) # File name file_name = file_path.name # Get text from CV resume_text = "/n".join([doc.page_content for doc in docs]) n_char = len(resume_text) n_words = count_words(resume_text) metadata={"file_name": file_name, "n_pages": n_pages, "n_char":n_char, "n_words":n_words} doc = Document(page_content=resume_text, metadata = metadata)
    
def load_pdf(file_path: Path) -> Document:
    start = time.perf_counter()

    # Load data
    loader = PyMuPDFLoader(file_path)

    # List of documents
    docs = loader.load()

    # Total pages CV
    n_pages = len(docs)

    # File name
    file_name = file_path.name

    # Get text from CV
    resume_text = "\n".join(doc.page_content for doc in docs)

    n_char = len(resume_text)
    n_words = count_words(resume_text)

    metadata = {
        "file_name": file_name,
        "n_pages": n_pages,
        "n_char": n_char,
        "n_words": n_words,
    }

    doc = Document(
        page_content=resume_text,
        metadata=metadata,
    )

    elapsed = time.perf_counter() - start
    logger.info(
        "[PyMuPDFLoader] PDF '%s' loaded in %.2f s",
        file_name,
        elapsed,
    )

    return doc
def load_folder(folder_path:Path)->list[Document]:
    docs = []
    for file in folder_path.glob("*.pdf"):
        docs.append(load_pdf(folder_path/file.name))
    return docs

from docling.document_converter import DocumentConverter

def extract_cv_with_docling(file_path: str | Path) -> str:
    start = time.perf_counter()

    converter = DocumentConverter()
    result = converter.convert(str(file_path))

    elapsed = time.perf_counter() - start
    logger.info(f"[Docling] CV analyzed in {elapsed:.2f} s")

    return result.document.export_to_markdown()

import pandas as pd

def create_dataset(docs:list[Document])->pd.DataFrame:
    df = pd.DataFrame([
    {
        "content": doc.page_content,
        **doc.metadata
    }
    for doc in docs
    ])
    return df

def load_all_results(output_folder: str) -> list[dict]:
    results = []
    for file_path in sorted(Path(output_folder).glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results

def clean_parsed_cv(text:str):
    patterns = ["<!-- image -->"]
    for pattern in patterns:
        text = text.replace(pattern,"").strip()
    return text

def process_cv(path:Path):

    with open(path, "r") as f:
        cv_text = f.read()

    cv_text = clean_parsed_cv(text = cv_text)
    return cv_text

def process_cvs_folder(folder_path: Path):
    cvs = {}

    for path in folder_path.iterdir():
        if path.is_file():
            cvs[path.name] = process_cv(path)

    return cvs


def process_cvs_to_dataframe(folder_path: Path):

    cvs = {
        path.stem: process_cv(path)
        for path in folder_path.glob("*.txt")
    }

    return pd.DataFrame(
        cvs.items(),
        columns=["id", "cv_text"]
    )

def load_all_results(output_folder: str) -> list[dict]:
    results = []
    for file_path in sorted(Path(output_folder).glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results

def return_chunk(chunk_id:str, key_column:str = "chunk_id", exp_column = "experience")->str:
    data = pd.read_json(DATA_FOLDER/"resumes_full.json")
    data_chunk = data[data[key_column] == chunk_id]
    return print(data_chunk[exp_column].iloc[0])

if __name__ == "__main__":
    
    print(PROJECT_ROOT)