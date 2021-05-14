import streamlit as st
import spacy
from name_generator.models.ly_model import LyModel

st.title('Business Namer')


st.write('Please describe your business')
description = st.text_input(label='Description')
st.write(description)

model_options = [
    '-ly model',
]

model_name = st.selectbox(
    label='Select model',
    options=model_options)

if model_name == model_options[0]:
    model = LyModel()
    nlp = spacy.load("en_core_web_sm")

if st.button('button'):
    if model_name == model_options[0]:
        doc = nlp(description)
        for chunk in doc.noun_chunks:
            st.write(model.predict(chunk.lemma_.split(' ')[0]).title())
