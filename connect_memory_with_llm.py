import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, RetrievalQA 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# SETUP MISTRAL
HF_TOKEN = os.environ.get("HF_TOKEN")
# get repo id hugging face 
HUGGING_FACE_REPO_ID='mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(HuggingFace_repoID):
    llm=HuggingFaceEndpoint (
        repo_id=HuggingFace_repoID,
        temperature=0.5,
        model_kwargs={'token':HF_TOKEN,
                      "max_length":512}
    )
    return llm

# CONNECT LLM WITH FAISS_DB AND CREATE CHAIN
# custom prompt(system prompt give answer and context, no hallucinate)


CUSTOM_PROMPT_TEMPLATE = """
Based on the provided context, answer the question as accurately as possible. 
Do not include any information not present in the context. 
If the context does not contain the answer, respond with: "I don't know based on the provided context."

Context: {context}

Question: {question}

start answer directly no small talk
"""



def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])                             # 
    return prompt
# prompt created amd database path we know, load model with databse

DB_PATH = 'vectorstore/DB_FAISS'
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_PATH,embedding_model,allow_dangerous_deserialization=True)

# CREATE QA CHAIN(conversational/retrieval)
# look into lancgain doc chain types
# combine_docs_chain = create_stuff_documents_chain(
#     llm=load_llm(HuggingFace_repoID),
#     prompt=set_custom_prompt
# )


# qa_chain = create_retrieval_chain(
#     retriever= db.as_retriever(search_kwargs={'k':4}), # how many docs should be similar(top how many) cause ranked result from retriveal from db
#     return_source_documents= True , # metadata like pagenumber
#     combine_docs_chain=combine_docs_chain
# )

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGING_FACE_REPO_ID),
    chain_type='stuff',
    retriever= db.as_retriever(search_kwargs={'k':1}), # how many docs should be similar(top how many) cause ranked result from retriveal from db
    return_source_documents= True , # metadata like pagenumber
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# invoke chain with single query
user_query=input('write query here : ')
response=qa_chain.invoke({'query':user_query})
print('Result:',response['result'])
print('Source Documents: ',response['source_documents'])