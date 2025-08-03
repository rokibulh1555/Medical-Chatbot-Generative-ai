from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
pc = Pinecone(api_key= PINECONE_API_KEY)

# if not pc.has_index(index_name):
#     pc.create_index_for_model(
#         name=index_name,
#         dimension = 384,
#         metric = "cosine",
#         spec = ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name= index_name,
    embedding= embeddings
)