# import packages
from dotenv import load_dotenv
from openai import OpenAI
import os
import streamlit as st

# load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_data
def get_response(user_prompt, temperature):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_output_tokens=100
    )
    return response


st.title("Generative AI App")
st.write("This app uses OpenAI's model")

user_prompt = st.text_input("Enter your prompt here:")

temperature = st.slider(
    "Select temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative")


with st.spinner("AI is working..."):
    response = get_response(user_prompt, temperature)
    # print the response from OpenAI
    st.write(response.output_text)
