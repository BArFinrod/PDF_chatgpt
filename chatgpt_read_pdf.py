from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

import os
import streamlit as st
import pandas as pd
from io import StringIO

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PIL import Image


## Logo
image = Image.open(Path(__file__).parent / 'Logo_blanco.jpeg')

token = st.text_input('Token', 'Inserte aquí su token')

os.environ["OPENAI_API_KEY"] = token

uploaded_file = st.file_uploader("Elija un archivo PDF")

if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    reader = PdfReader(uploaded_file)

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # query = "¿a cuánto ascendieron los despachos de cemento entre enero y diciembre?"
    query = st.text_input('Pregunta', 'Inserte aquí su pregunta')

    if query!='Inserte aquí su pregunta':
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write('Respuesta: ', answer)
    else:
        st.write('Respuesta: ', 'Aún no ha preguntado nada')
