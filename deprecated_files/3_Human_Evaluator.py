#---------------- Imports ----------------#
import streamlit as st
import pandas as pd 
import module
import os
import time
from datetime import date, datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

# from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate

from streamlit_option_menu import option_menu

from google.cloud import firestore

#---------------- Page Setup ----------------#
page_title = "Human Evaluator"
page_icon = "👤"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Human Evaluator'*")

#---------------- Sidebar ----------------#
with st.sidebar:
    st.write("## Settings")
    oai_key = st.selectbox("Select OpenAI Key", ["RPI", "Personal"])
    if oai_key == "Personal":
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_personal"]
    else:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_team"]
    "---"

#---------------- Common Variables / Functions ----------------#
db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

###################################################
#                                                 #
#               Human as Evaluator                #
#                                                 #
###################################################

st.write("Coming Soon...")


#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")