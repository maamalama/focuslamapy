from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import Dict
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from flask_cors import CORS

import chromadb

app = Flask(__name__)
CORS(app)

llm = Groq(model="llama3-8b-8192", pydantic_program_mode="llm", api_key="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")
#Ollama(model="llama3")
class Website(BaseModel):
    """A website category."""
    category: str

def connect_db():
    return psycopg2.connect(host='aws-0-us-west-1.pooler.supabase.com', dbname='aws-0-us-west-1.pooler.supabase.com', user='postgres.yokdugrvbixeiyhkgqtc', password='kyGXn22Bxlh8jpYL')

def store_embedding(vector):
    with connect_db() as conn:
        with conn.cursor() as cur:
            # Assuming vector is a NumPy array and needs to be converted to list for pgvector
            cur.execute("INSERT INTO embeddings (vector) VALUES (%s)", [vector.tolist()])
            conn.commit()
            print("Embedding stored successfully.")

@app.route('/classify', methods=['POST'])
def classify_event():
    try:

        # Classify the event using LlamaIndex
        
        prompt_tmpl = PromptTemplate(
            "Determine website category based on website utl: {content}."
        )
        print(prompt_tmpl)
        website_category = llm.structured_predict(
            Website, prompt_tmpl, content=request.json["content"]
        )
        print(website_category)
        return jsonify(website_category.dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

class Distraction(BaseModel):
    """A distraction level between 0 and 10."""
    distraction_level: int

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    try:

        # Classify the event using LlamaIndex
        
        prompt_tmpl = PromptTemplate(
            "Imagine you're an expert at analyzing human behavior. Based on web site traffic data, rate the degree of distraction from work from 0 to 10. Make sure that your answer is correct depending on the sites visited and their number, for example, if a person visited for a minute 3 times reddit, 3 times youtube and 1 time site related to non-entertainment (different categories), he clearly began to be distracted, if it's primary only 1-2 same frequent websites non distraction category like Other, Work, Tech and etc then it's not a distraction. Data from recent visits: {content}."
        )
        print(prompt_tmpl)
        distraction_level = llm.structured_predict(
            Distraction, prompt_tmpl, content=request.json["content"]
        )
        print(distraction_level)
        return jsonify(distraction_level.dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/embeddings', methods=['POST'])
def embeddings():
    try:
        content = request.json["content"]
        ollama_embedding = OllamaEmbedding(
            model_name="llama2",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )

        pass_embedding = ollama_embedding.get_text_embedding_batch(
            [content], show_progress=True
        )
        documents = [
            {"content": content, "embedding": pass_embedding[0]}
        ]
        index = VectorStoreIndex.from_documents(documents)
        print(restaurant_obj)
        return jsonify(restaurant_obj.dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == '__main__':
    app.run(debug=True)