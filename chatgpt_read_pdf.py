from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from pathlib import Path

import os
import streamlit as st
import pandas as pd
from io import StringIO

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PIL import Image


## Logo
image = Image.open(Path(__file__).parent / 'Logo_blanco.jpeg')
st.image(image, width=150)

st.markdown('El código fue obtenido de [YouTube](https://www.youtube.com/watch?v=TLf90ipMzfE&t=434s) e implementado en esta aplicación de Streamlit.', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Instrucciones', unsafe_allow_html=True)
st.markdown('<p class="medium-font">1. Obtenga un token de forma gratuita en la siguiente dirección: [OpenIA](https://platform.openai.com/account/billing/overview) . Es posible que luego de usarlo varias veces en esta aplicación, el token quede obsoleto \
debido a que se acabó la prueba gratuita o por motivos de prevención de OpenIA. No es motivo de preocuparse ya que puede generarse otro token. Esta aplicación ha sido realizada solo con fines demostrativos. Tal vez pueda implementar su propia solución \
siguiendo los pasos del video de YouTube indicado en el punto 1', unsafe_allow_html=True)
st.markdown('<p class="medium-font">2. Ingrese el token en la siguiente espacio indicado. Esta aplicación no almacena tokens ni ningún tipo de información: De hecho no tengo espacio para ello', unsafe_allow_html=True)
st.markdown('<p class="medium-font">3. Suba su archivo PDF y comience a hacer preguntas sobre el mismo.', unsafe_allow_html=True)

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
