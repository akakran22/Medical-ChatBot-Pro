import os
import re
import fitz  

class DocumentProcessor:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.pdf_files = [
            "Standard_Treatment_Guidelines.pdf",
            "The_Gale_Encyclopedia_Of_Medicine.pdf"
        ]

    def clean_text(self, text):
        #Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
        text = re.sub(r'\f', ' ', text)
        text = re.sub(r'\.{3,}', '...', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_all_documents(self):
        #Get all document texts
        documents = []
        
        for pdf_file in self.pdf_files:
            pdf_path = os.path.join(self.data_folder, pdf_file)
            if os.path.exists(pdf_path):
                print(f"Processing {pdf_file}...")
                try:
                    doc = fitz.open(pdf_path)
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        
                        if text.strip():  # Only add non-empty pages
                            # Create a document-like object
                            class SimpleDoc:
                                def __init__(self, content, metadata):
                                    self.page_content = content
                                    self.metadata = metadata
                            
                            cleaned_text = self.clean_text(text)
                            if cleaned_text:
                                simple_doc = SimpleDoc(
                                    cleaned_text,
                                    {
                                        'source': pdf_file,
                                        'page': page_num + 1
                                    }
                                )
                                documents.append(simple_doc)
                    
                    doc.close()
                    print(f"Loaded {pdf_file}: {len([d for d in documents if d.metadata['source'] == pdf_file])} pages")
                    
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")
            else:
                print(f"File not found: {pdf_path}")
        
        return documents