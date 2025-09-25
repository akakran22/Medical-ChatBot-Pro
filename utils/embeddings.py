import os
import numpy as np
from typing import List
from tqdm import tqdm
import cohere
import time

class EmbeddingManager:
    #Embeddings via cohere api
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY required")
        
        self.client = cohere.Client(api_key)
        self.embedding_dim = 384

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        #Generate embeddings for multiple texts
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"Processing {len(texts)} texts...")
        
        all_embeddings = []
        batch_size = 10  # Reduced batch size to avoid rate limits
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            clean_batch = [text.strip()[:1500] if text.strip() else "empty" for text in batch]  # Reduced text length
            
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    response = self.client.embed(
                        texts=clean_batch,
                        model="embed-english-light-v3.0",
                        input_type="search_document"
                    )
                    batch_embeddings = response.embeddings
                    all_embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "rate limit" in error_str.lower():
                        wait_time = 60  # Wait 1 minute for rate limit
                        print(f"\nRate limit hit. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"Batch error: {e}")
                        dummy_embeddings = [[0.0] * self.embedding_dim for _ in batch]
                        all_embeddings.extend(dummy_embeddings)
                        break
            
            if retry_count >= max_retries:
                print(f"Max retries reached for batch {i}. Using dummy embeddings.")
                dummy_embeddings = [[0.0] * self.embedding_dim for _ in batch]
                all_embeddings.extend(dummy_embeddings)
            
            # Rate limiting between batches
            time.sleep(2)  # Wait 2 seconds between batches
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings

    def get_query_embedding(self, query: str) -> np.ndarray:
        #Generate embedding for a single query
        try:
            response = self.client.embed(
                texts=[query.strip()[:1500]],
                model="embed-english-light-v3.0",
                input_type="search_query"
            )
            embedding = np.array(response.embeddings[0], dtype=np.float32)
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("Rate limit hit for query. Waiting 30 seconds...")
                time.sleep(30)
                try:
                    response = self.client.embed(
                        texts=[query.strip()[:1500]],
                        model="embed-english-light-v3.0",
                        input_type="search_query"
                    )
                    embedding = np.array(response.embeddings[0], dtype=np.float32)
                    return embedding / np.linalg.norm(embedding)
                except:
                    pass
            
            print(f"Query embedding error: {e}")
            dummy = np.random.rand(self.embedding_dim).astype(np.float32)
            return dummy / np.linalg.norm(dummy)