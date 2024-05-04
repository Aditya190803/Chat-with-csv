from pandasai.llm.local_llm import LocalLLM
import streamlit as st 
import pandas as pd 
from pandasai import SmartDataframe
import os
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatGroq(model_name="llama3-70b-8192",api_key = os.environ["GROQ_API_KEY"])
#model = LocalLLM(api_base="http://localhost:11434/v1",model="llama2")

st.title("Data analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": model})
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))

