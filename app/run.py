import logging
from pathlib import Path

import streamlit as st
from name_generator.models.encoder_decoder_model import EncoderDecoderModel
from name_generator.models.ly_model import LyModel


@st.cache
def get_logger_streamlit():
    file_handler = logging.FileHandler(filename='./data/feedback.log')
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    return logger


logger = get_logger_streamlit()

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
elif model_name == model_options[1]:
    model_path = st.selectbox(
        label='Select model version',
        options=model_list,
        format_func=lambda x: x.name)
    model = EncoderDecoderModel(model_path)

if st.button('Generate name'):
    prediction = model.predict(description)
    st.write(prediction)
    logger.info(f'Description: {description}')
    logger.info(f'Prediction: {prediction}')

    if st.button('Like'):
        logger.info('Feedback: Like')

    if st.button('DisLike'):
        logger.info('Feedback: DisLike')
