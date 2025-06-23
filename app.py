from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os
#import google.generativeai as genai
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAI
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser



st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    

st.title("PDF READER")

#uploaded_file = st.file_uploader("Chose a file")
 

def get_pdf_text(pdf_docs):
    text = ""
  #  if len(pdf_docs) > 1:
    for pdf in pdf_docs:  # file is each uploaded PDF
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
                text += page.extract_text()
        return text        

  

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context,make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """ 

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embedding_model = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    
    new_db = FAISS.load_local("faiss_index", embedding_model,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")
    
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                raw_text = get_pdf_text(pdf_docs) # get the pdf text
                text_chunks = get_text_chunks(raw_text) # get the text chunks
                get_vector_store(text_chunks) # create vector store
                st.success("Done")
        
        st.write("---")


if __name__ == "__main__":
    main()