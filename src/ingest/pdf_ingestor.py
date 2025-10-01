import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load env variables (like OpenAI key if needed later)
load_dotenv()


class PDFIngestor:
    def __init__(self, base_path: str = "data", chunk_size: int = 800, chunk_overlap: int = 200):
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_pdfs(self) -> List[Path]:
        """Get list of all PDFs inside data/ folder (recursive)."""
        return list(self.base_path.glob("**/*.pdf"))

    def process_pdf(self, pdf_path: Path):
        """Load + split one PDF into chunks."""
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        return chunks

    def ingest(self) -> List:
        """Ingest all PDFs in base_path and return all chunks."""
        all_chunks = []
        pdfs = self.load_pdfs()
        print(f"Found {len(pdfs)} PDFs under {self.base_path}")

        for pdf in pdfs:
            chunks = self.process_pdf(pdf)
            print(f"→ {pdf.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

        return all_chunks


if __name__ == "__main__":
    ingestor = PDFIngestor(base_path="data")
    chunks = ingestor.ingest()
    print(f"\n✅ Total chunks created: {len(chunks)}")
    print("Sample chunk:\n", chunks[0].page_content[:500])
