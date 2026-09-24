# santhoshkumar R's AI Assistant 🤖

A RAG-powered chatbot that answers questions about santhoshkumar R's background, skills, and projects.


## Built With
- **Streamlit** - Web interface
- **FAISS** - Vector search
- **Groq** - LLM inference (openai/gpt-oss-120b)
- **fastembed** - Lightweight embeddings (ONNX, no torch)

## How It Works
1. Documents are chunked and embedded
2. User query is embedded
3. Relevant chunks are retrieved via vector similarity
4. LLM generates answer based on retrieved context

## Setup
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.streamlit/secrets.toml`
4. Run: `streamlit run streamlit_app.py`
