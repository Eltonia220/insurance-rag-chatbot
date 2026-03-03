from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

client = genai.Client(api_key="YOUR_GEMINI_API_KEY_HERE")

loader = PyPDFLoader("insurance_policy.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

def ask(question):
    results = vectorstore.similarity_search(question, k=3)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Using this context:\n{context}\n\nAnswer this question: {question}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

print(ask("What does this policy cover?"))
