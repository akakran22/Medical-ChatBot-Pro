import os
from typing import List, Dict
from tavily import TavilyClient

class WebScraper:    
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.medical_sites = [
            "mayoclinic.org",
            "webmd.com", 
            "medlineplus.gov",
            "healthline.com",
            "who.int",
            "cdc.gov",
            "nih.gov",
            "pubmed.ncbi.nlm.nih.gov"
        ]
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search web for relevant medical information"""
        try:
            # Add medical context to query
            medical_query = f"medical health {query} symptoms treatment diagnosis"
            
            response = self.client.search(
                query=medical_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=self.medical_sites
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })
            
            return results
            
        except Exception as e:
            print(f"Error in web search: {str(e)}")
            return []