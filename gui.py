import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load DB in cache
DB_PATH='vectorstore/DB_FAISS'
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None

# Prompt
def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])                             # 
    return prompt


# HF-TOKEN for load llm
def load_llm(HuggingFace_repoID,HF_TOKEN):
    llm=HuggingFaceEndpoint (
        repo_id=HuggingFace_repoID,
        temperature=0.5,
        model_kwargs={'token':HF_TOKEN,
                      "max_length":512}
    )
    return llm

def main():
    st.title('Ask Chatbot')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Pass your prompt here ')
    
    # Making Response  
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Based on the provided context, answer the question as accurately as possible. 
        Do not include any information not present in the context. 
        If the context does not contain the answer, respond with: "I don't know based on the provided context."

        Context: {context}

        Question: {question}

        start answer directly no small talk
        """
    
        REPO_ID='mistralai/Mistral-7B-Instruct-v0.3'
        HF_TOKEN = os.environ.get("HF_TOKEN")
        # llm=load_llm(REPO_ID,HF_TOKEN)
        res_to_show = "Fallback error"

        try:
            vector_store=get_vectorstore()
            if vector_store is None:
                st.error('Failed to load DB')

            # Create Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HuggingFace_repoID=REPO_ID,HF_TOKEN=HF_TOKEN),
                chain_type='stuff',
                retriever= vector_store.as_retriever(search_kwargs={'k':1}), # how many docs should be similar(top how many) cause ranked result from retriveal from db
                return_source_documents= True , # metadata like pagenumber
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result= response['result']
            source_doc=response['source_documents']
            res_to_show=result+'\nSource Docs: \n'+str(source_doc)

            st.chat_message('ai').markdown(res_to_show)
            st.session_state.messages.append({'role':'ai','content':res_to_show})

        except Exception as e:
            st.error(f'Error: {str(e)}')


            #response='Hi, i am AImedico!'
            
if __name__=='__main__':
    main()
