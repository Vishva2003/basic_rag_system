
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path


try:
    from ..config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
                            path=str(VECTOR_DB_PATH),
                            settings=Settings(anonymized_telemetry=False)
                            )

    def create_collection(self, collection_name):
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f'Using exixting collection: {collection_name}')
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f'Created new collection: {collection_name}')

    def embed_texts(self, text):
        embedded_text = self.model.encode(text)
        return embedded_text.tolist()

    def add_collection(self, text, collection_name, ids=None, metadatas=None):
        embeddings = self.embed_texts(text)

        if ids is None:
            ids = [f'doc{str(i)}' for i in range(len(text))]

        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(text))]

        self.collection.add(
            embeddings=embeddings,
            documents=text,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Added {len(text)} documents to collection: {collection_name}")


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker

    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)
    embedder = Embedder()
    embedder.create_collection('beta_vae_collection')

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"
    if pdf_path.exists():
        text = loader.load_doc(str(pdf_path))

        # Chunk it
        chunks = chunker.chunk_text(text)

        print(f"loaded document with {len(text)} characters")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk ({len(chunks[0])} chars):")
        embedder.add_collection(chunks, 'beta_vae_collection')
