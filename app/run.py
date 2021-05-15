from pathlib import Path
import streamlit as st
import torch
from name_generator.models.encoder_decoder_model import EncoderDecoderModel
from name_generator.models.ly_model import LyModel
from name_generator.models.nn_models import AttnDecoderRNN, EncoderRNN

st.title('Business Namer')


st.write('Please describe your business')
description = st.text_input(label='Description')
st.write(description)


models_path = Path('./data/output/models')
model_list = list(models_path.iterdir())


model_options = [
    '-ly model',
    'seq-to-char model'
]

model_name = st.selectbox(
    label='Select model type',
    options=model_options)

if model_name == model_options[0]:
    model = LyModel()
    nlp = spacy.load("en_core_web_sm")
elif model_name == model_options[1]:
    model_path = st.selectbox(
        label='Select model version',
        options=model_list,
        format_func=lambda x: x.name)
    model = EncoderDecoderModel(model_path)

if st.button('button'):
    if model_name == model_options[0]:
        doc = nlp(description)
        for chunk in doc.noun_chunks:
            st.write(model.predict(chunk.lemma_.split(' ')[0]).title())
    elif model_name == model_options[1]:
        st.write(model.predict(description))
