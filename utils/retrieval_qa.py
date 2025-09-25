import os
from typing import List, Dict
from groq import Groq

class LLMAgent:
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def generate_response(self, query: str, vector_context: List[Dict], 
                         web_context: List[Dict]) -> str:
        try:
            # Handle simple greetings
            q_lower = query.strip().lower()
            if q_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
                return "Hello! I'm here to help you with medical questions. How can I assist you today?"
            
            if q_lower in ["how are you", "how is it going", "what's up", "how do you do"]:
                return "I'm ready to help you with medical information. What would you like to know?"
            
            # Prepare context from vector database
            vector_text = ""
            if vector_context:
                vector_text = "\n\n".join([
                    f"Medical Document - {item['source']} (Page {item.get('page', 'Unknown')}):\n{item['text'][:800]}..."
                    for item in vector_context[:3]
                ])
            
            # Prepare context from web search
            web_text = ""
            if web_context:
                web_text = "\n\n".join([
                    f"Web Source - {item['title']}:\n{item['content'][:800]}..."
                    for item in web_context[:3]
                ])
            
            # System prompt for plain text without markdown/dashes
            system_prompt = """You are a medical AI assistant that provides accurate, helpful medical information. 
You have access to medical literature and current web information.

FORMATTING RULES:
- Do NOT use Markdown (#, *, **, etc.)
- Do NOT use dashes (-) or underlines (---)
- For lists, only use numbering (1., 2., 3.) or letters (a., b., c.)
- For headings, just write them in normal sentence case (e.g., "Understanding Asthma Triggers")
- Keep the output as clean plain text with paragraphs and numbered/lettered lists only

CONTENT RULES:
1. Provide comprehensive, evidence-based medical information
2. Combine medical literature and current web sources where possible
3. Explain medical terms in simple language
4. Always include a medical disclaimer at the end
5. Be empathetic and professional
"""

            # Combine contexts
            context_section = ""
            if vector_text:
                context_section += f"MEDICAL LITERATURE:\n{vector_text}\n\n"
            if web_text:
                context_section += f"CURRENT WEB INFORMATION:\n{web_text}\n\n"
            
            if not context_section:
                context_section = "No specific context found. Providing general medical knowledge response.\n\n"

            user_prompt = f"""Medical Query: {query}

AVAILABLE CONTEXT:
{context_section}

Please provide a comprehensive plain-text response that:
1. Answers the query directly
2. Uses numbered or alphabetic lists (1., 2., 3. or a., b., c.)
3. Avoids markdown, underlines, and dashes
4. Explains complex terms simply
5. Includes a disclaimer and suggests consulting a doctor if needed

Response:"""

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                max_tokens=1200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm unable to generate a response at this time. Please try again later, or consult with a healthcare professional for medical advice."
