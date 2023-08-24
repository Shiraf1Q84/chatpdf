# Import libraries
import os
import tempfile
import streamlit as st

from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_api_and_file():
    user_api_key = st.sidebar.text_input(
        label="OpenAI API key",
        placeholder="Paste your openAI API key here",
        type="password"
    )
    uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
    os.environ['OPENAI_API_KEY'] = user_api_key
    return user_api_key, uploaded_file

def load_pdf(uploaded_file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=len)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(file_path=tmp_file_path)
    return loader.load_and_split(text_splitter)

def setup_chain(data):
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
        retriever=vectors.as_retriever()
    )
    return chain

def conversational_chat(chain, query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_state(uploaded_file):
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! Feel free to ask about anything regarding this" + uploaded_file.name])
    st.session_state.setdefault('past', ["Hi!"])

def chat_interface(chain):
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Input:", placeholder="Please enter your message regarding the PDF data.", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(chain, user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i, (past_msg, generated_msg) in enumerate(zip(st.session_state["past"], st.session_state["generated"])):
                message(past_msg, is_user=True, key=f'{i}_user', avatar_style="big-smile")
                message(generated_msg, key=str(i), avatar_style="thumbs")

def main():
    user_api_key, uploaded_file = get_api_and_file()

    if uploaded_file:
        data = load_pdf(uploaded_file)
        chain = setup_chain(data)
        initialize_state(uploaded_file)
        chat_interface(chain)

if __name__ == '__main__':
    main()
