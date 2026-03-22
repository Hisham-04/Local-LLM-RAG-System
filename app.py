from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Enterprise RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def root():
    return FileResponse("static/index.html")
device = "cuda"
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')

client = chromadb.PersistentClient(path='/home/hisham/enterprise-rag/chroma_db')
collection = client.get_or_create_collection('enterprise_docs')

def generate_with_qwen(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors='pt').to(device)
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None
    )
    output = output[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output, skip_special_tokens=True)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    question_embedding = embedding_model.encode(
        ["query: " + q.question]
    ).tolist()
    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    context = '\n\n'.join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))

    messages = [
        {
            'role': 'system',
            'content': '\n'.join([
                'You are an enterprise HR assistant for GitLab.',
                'Answer questions based only on the provided context.',
                'If the answer is not in the context, say: I do not have information about that, please contact HR.',
                'Be concise and professional.'
            ])
        },
        {
            'role': 'user',
            'content': f'Context:\n{context}\n\nQuestion: {q.question}\n\nAnswer:'
        }
    ]

    response = generate_with_qwen(messages)
    
    return {
        "answer": response,
        "sources": sources
    }

@app.get("/health")
def health():
    return {"status": "ok"}
