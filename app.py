from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import Dict
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq

app = Flask(__name__)

llm = Groq(model="llama3-8b-8192", pydantic_program_mode="llm", api_key="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")
#Ollama(model="llama3")
class Website(BaseModel):
    """A website category."""
    category: str

@app.route('/classify', methods=['POST'])
def classify_event():
    try:

        # Classify the event using LlamaIndex
        
        prompt_tmpl = PromptTemplate(
            "Determine website category based on website utl: {content}."
        )
        print(prompt_tmpl)
        restaurant_obj = llm.structured_predict(
            Website, prompt_tmpl, content=request.json["content"]
        )
        print(restaurant_obj)
        return jsonify(restaurant_obj.dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)