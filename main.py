import pandas as pd
import os
import streamlit as st
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from pandasai.responses.response_parser import ResponseParser

# Initialize the language model
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=os.environ["GROQ_API_KEY"])

def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

class StreamlitResponse(ResponseParser):
    def format_dataframe(self, result):
        st.dataframe(result["value"])

    def format_plot(self, result):
        st.image(result["value"])

    def format_other(self, result):
        st.write(result["value"])

# Streamlit app title
st.write("# Chat with CSV Data 🦙")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Create a container for chat history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write(df)

    # Form to capture user input and submit
    query = st.text_input("🗣️ Ask your question:")

    # Process the query when the user submits
    if query:
        # Add user query to the chat history
        st.session_state.history.append({"user": query})

        with st.spinner("Generating response..."):
            query_engine = SmartDataframe(df, config={"llm": llm, "response_parser": StreamlitResponse})
            answer = query_engine.chat(query)

            # Add bot response to the chat history
            st.session_state.history.append({"bot": answer})

        # Display the chat history
        for message in st.session_state.history:
            if "user" in message:
                st.markdown(f"**You:** {message['user']}")
            if "bot" in message:
                st.markdown(f"**Bot:** {message['bot']}")

        # Clear the text input after processing the query
        st.text_input("🗣️ Ask your question:", value="", key="query_input", disabled=False)
