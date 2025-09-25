import os
import json
from typing import List, Dict
from groq import Groq

class CriticAgent:
    """Evaluates response quality and decides if more information is needed"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def evaluate_response(self, query: str, response: str, 
                         vector_context: List[Dict], web_context: List[Dict]) -> Dict:
        """Evaluate response quality and provide score"""
        try:
            critic_prompt = f"""You are a medical response critic. Evaluate the quality of this medical response on a scale of 1-10.

USER QUERY: {query}

RESPONSE TO EVALUATE: {response}

AVAILABLE CONTEXT:
- Vector DB results: {len(vector_context)} medical documents
- Web search results: {len(web_context)} articles

Evaluate based on:
1. Medical accuracy (30%)
2. Completeness of answer (25%)
3. Clarity and understandability (20%)
4. Appropriate use of context (15%)
5. Proper medical disclaimers (10%)

Provide your evaluation as a JSON object:
{{
    "score": <score_1_to_10>,
    "reasoning": "<brief_explanation>",
    "needs_more_info": <true/false>,
    "suggestions": "<improvement_suggestions>"
}}
"""

            response_eval = self.client.chat.completions.create(
                messages=[{"role": "user", "content": critic_prompt}],
                model=self.model,
                max_tokens=500,
                temperature=0.1
            )
            
            try:
                eval_result = json.loads(response_eval.choices[0].message.content)
                return eval_result
            except:
                return {
                    "score": 7.0,
                    "reasoning": "Could not parse evaluation",
                    "needs_more_info": False,
                    "suggestions": "Response appears adequate"
                }
                
        except Exception as e:
            print(f"Error in critic evaluation: {str(e)}")
            return {
                "score": 5.0,
                "reasoning": "Evaluation failed",
                "needs_more_info": False,
                "suggestions": "Unable to evaluate"
            }