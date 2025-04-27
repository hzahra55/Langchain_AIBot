from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



# LOAD RAW PDF 
DATA_PATH= "data/"
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob='**/*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents= loader.load()
    return documents

import os
print(os.listdir(DATA_PATH))
documents=load_pdf(data=DATA_PATH)
print("len doc", len(documents))

# CREATE CHUNKS
def create_chunks(extracted_books):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=60)
    text_chunks=text_splitter.split_documents(extracted_books)
    return text_chunks

text_chunks = create_chunks(documents)
print('no of text chunks',len(text_chunks))


# CREATE VECTOR EMBEDDINGS
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model


# STORE EMBEDDINGS IN FAISS
DB_PATH= 'vectorstore/DB_FAISS'
embedding_model=get_embedding_model()
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_PATH)
