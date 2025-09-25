from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class TextChunker:    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "Section ", "SECTION ", "Sec. ", "CHAPTER ", "Chapter ", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List) -> List[Dict[str, Any]]:
        #Chunk docs into smaller pieces
        chunked_documents = []
        
        for doc in documents:
            # Handle both dict format and document objects
            if hasattr(doc, 'page_content'):
                text = doc.page_content
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", 0)
                metadata = doc.metadata
            else:
                text = doc["text"]
                source = doc.get("source", "unknown")
                page = doc.get("page", 0)
                metadata = doc
            
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "id": f"{source}_{page}_{i}",
                    "text": chunk,
                    "source": source,
                    "page": page,
                    "chunk_id": i,
                    **metadata
                })
        
        return chunked_documents