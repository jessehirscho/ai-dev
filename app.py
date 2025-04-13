from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(directory=".", path="index.html")

# Load PDF
loader = PyPDFLoader("test.pdf")  # replace with your actual file name
documents = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ✅ Use Hugging Face embeddings (free, local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Optional: if you still want OpenAI for answers (can swap this too if needed)
qa_chain = RetrievalQA.from_chain_type(
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 100}
),  # still uses OpenAI unless swapped
    retriever=vectorstore.as_retriever()
)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    response = qa_chain.run(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)