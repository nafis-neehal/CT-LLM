#---------------- Imports ----------------#
import streamlit as st
import pandas as pd 
# import module
# import os
# import time
# from datetime import date, datetime, timezone

# from langchain_openai import ChatOpenAI
# from langchain_community.llms import HuggingFaceEndpoint

# # from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain.prompts.prompt import PromptTemplate
# from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate

# from streamlit_option_menu import option_menu

from google.cloud import firestore

#---------------- Page Setup ----------------#

page_title = "CTBench"
page_icon = "🧪" #"🤖" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Welcome to CTBench: A Comprehensive Benchmark for Clinical Trial Design Automation Tasks'*")
st.image("images/welcome_img.png", caption="Image Credit: bsd studio/Shutterstock.com", use_column_width=True)


#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")

