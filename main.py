from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser

# Custom response parser to display outputs in Streamlit
class StreamlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)  # Pass context to the parent class

    def format_dataframe(self, result):
        st.dataframe(result["value"])

    def format_plot(self, result):
        st.image(result["value"])  # Display plot image

    def format_other(self, result):
        st.write(result["value"])

# Initialize the model
try:
    model = GoogleGenerativeAI(api_key=st.secrets["GOOGLE_API_KEY"], model="gemini-2.0-flash")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")

st.title("Data Analysis with PandaAI")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", data.head(3))

    try:
        df = SmartDataframe(data, config={
            "llm": model,
            "response_parser": StreamlitResponse,  # Pass the class, not an instance
            "enforce_privacy": False
        })
        st.success("SmartDataframe initialized.")
    except Exception as e:
        st.error(f"Failed to initialize SmartDataframe: {e}")

    prompt = st.text_area("🔎 Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    response = df.chat(prompt)
                    if response:  # Handles non-plotted plain text or structured responses
                        st.write(response)
                except Exception as e:
                    st.error(f"Error during generation: {e}")
