import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pinecone
from utlis import read_doc, chunk_data

# Load environment variables
load_dotenv()

# Read and chunk documents
documents = read_doc('Documents/')
chunked_documents = chunk_data(docs=documents)
print(f"Total Chunks Created: {len(chunked_documents)}")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment='us-east1-gcp'
)

# Define Pinecone Index Name
index_name = "langchainvector"

# Check if index exists, if not, create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # Dimension of OpenAI's embeddings
        metric="cosine"
    )

# Connect to the index
index = pinecone.Index(index_name)

# Initialize Pinecone Vector Store
vector_store = Pinecone(index, embeddings.embed_query, "text")

# Add documents to the vector store
texts = [doc.page_content for doc in chunked_documents]
metadatas = [doc.metadata for doc in chunked_documents]
vector_store.add_texts(texts=texts, metadatas=metadatas)

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create Retrieval-based QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# Function to Get LLM Response
def get_response(query):
    response = qa_chain.run(query)
    return response

# Example Query
query = "What is the main topic of the documents?"
response = get_response(query)

print("\n### LLM Response: ###\n")
print(response)
