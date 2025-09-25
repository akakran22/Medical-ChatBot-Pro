import os
import time
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .embeddings import EmbeddingManager

class VectorDatabase:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Medical")
        self.embedding_manager = EmbeddingManager()
        self.vector_size = 384 

    def check_collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            print(f"Error checking collection: {e}")
            return False

    def get_collection_count(self) -> int:
        try:
            if self.check_collection_exists():
                info = self.client.get_collection(self.collection_name)
                return info.points_count
            return 0
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0

    def reset_collection(self):
        #Delete and recreate collection
        try:
            if self.check_collection_exists():
                print(f"Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
                time.sleep(2)
            
            print(f"Creating new collection with {self.vector_size} dimensions...")
            return self.create_collection()
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False

    def create_collection(self):
        #Create collection if missing
        try:
            if self.check_collection_exists():
                try:
                    info = self.client.get_collection(self.collection_name)
                    if info.points_count > 0:
                        print(f"Collection '{self.collection_name}' already exists with {info.points_count} documents")
                        return True
                    else:
                        print(f"Collection '{self.collection_name}' exists but is empty")
                except Exception as e:
                    print(f"Error checking collection details: {e}")
                    print("Deleting existing collection...")
                    self.client.delete_collection(self.collection_name)
                    time.sleep(2)
            
            print(f"Creating collection with {self.vector_size} dimensions...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created with {self.vector_size} dimensions")
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            print(f"Storing {len(documents)} documents...")
            
            # Get embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = self.embedding_manager.get_embeddings(texts)

            # Prepare points
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid.uuid4())
                
                # Ensure embedding is the right format
                if hasattr(embedding, 'tolist'):
                    vector = embedding.tolist()
                else:
                    vector = list(embedding)
                
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": doc["text"],
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0),
                        "chunk_id": doc.get("chunk_id", i),
                        "doc_id": doc.get("id", i)
                    }
                )
                points.append(point)

            # Upload in batches
            batch_size = 100
            successful = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    successful += len(batch)
                    print(f"Uploaded batch {i//batch_size + 1}: {successful}/{len(points)} points")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Batch upload error: {e}")
                    continue
            
            print(f"Successfully stored {successful}/{len(points)} documents")
            return successful > 0
            
        except Exception as e:
            print(f"Error storing documents: {e}")
            return False

    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedding_manager.get_query_embedding(query)
            
            if hasattr(query_embedding, 'tolist'):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "score": float(r.score),
                    "page": r.payload.get("page", 0),
                    "chunk_id": r.payload.get("chunk_id", -1)
                }
                for r in results
            ]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []