import streamlit as st
import os, re, json, time, math, tempfile
import nltk
import pdfminer.high_level
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.graph_objects as go

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="Personality Predictor | HEX Softwares",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
