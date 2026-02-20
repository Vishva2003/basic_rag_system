from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path
try:
    from ..config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW


class Retriever:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
                            path=str(VECTOR_DB_PATH),
                            settings=Settings(anonymized_telemetry=False)
                            )
        self.top_k = TOP_K_RESULTS
        self.collection = None

    def retrieve(self, query, collection_name):

        try:
            self.collection = self.client.get_collection(collection_name)
            print(f'Connected to collection: {collection_name}')

        except:
            print(f"Collection {collection_name} not found.")
            return []

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K_RESULTS
        )

        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0] if 'metadatas' in results else None,
            'distances': results['distances'][0],
            'ids': results['ids'][0]
        }

    def format_context(self, retrieved_data):
        formatted_context = []

        for i, (doc, dist) in enumerate(zip(retrieved_data['documents'], retrieved_data['distances'])):
            formatted_context.append(f"Document {i+1} (Distance: {dist:.2f}):\n{doc}\n")
        return '\n'.join(formatted_context)


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker
    from embedder import Embedder

    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)
    embedder = Embedder()
    retriever = Retriever()

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"

    if pdf_path.exists():

        text = loader.load_doc(str(pdf_path))

        chunks = chunker.chunk_text(text)

        embedder.create_collection('beta_vae_collection')

        query = 'What is beta VAE?'

        print(f"loaded document with {len(text)} characters")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk ({len(chunks[0])} chars):")

        embedder.add_collection(chunks, 'beta_vae_collection')
        retrieved_data = retriever.retrieve(query, 'beta_vae_collection')
        context = retriever.format_context(retrieved_data)
        print(f"\nRetrieved Context:\n{context}")
