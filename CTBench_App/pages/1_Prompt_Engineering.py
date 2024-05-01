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
page_title = "Prompt Engineering"
page_icon = "🔧"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Prompt Engineering Playground'*")

#---------------- Sidebar ----------------#
with st.sidebar:
    st.write("## Settings")
    oai_key = st.selectbox("Select OpenAI Key", ["Personal", "RPI"])
    if oai_key == "Personal":
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_personal"]
    else:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_team"]
    "---"

#---------------- Common Variables / Functions ----------------#
db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

###################################################
#                                                 # 
#               Prompt Engineering                #
#                                                 #
###################################################

all_silver_ids = module.get_silverdata_ids()

# Default values for the system and human instructions -- fetch from DB
docs = db.collection('prompts').document('gpt-4-prompts').collection('all_prompts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
for doc in docs:
    default_system = doc.to_dict()['prompt']
    last_updated = module.format_firestore_timestamp(doc.to_dict()['timestamp'])

#---------------- Step 1: Current Query Trial ----------------#

# Initialize a session state for the current index
if 'index' not in st.session_state:
    st.session_state.index = 0

st.write(f"Trial {st.session_state.index + 1} of {len(all_silver_ids)}")

# Create next and previous buttons
c1, c2, _, _, _ = st.columns(5)
with c1:
    if st.button('◀ Previous'):
        # Decrement the index, ensuring it doesn't go below 0
        st.session_state.index = max(0, st.session_state.index - 1)
with c2:
    if st.button('Next ►'):
        # Increment the index, ensuring it doesn't go above the number of documents
        st.session_state.index = min(len(all_silver_ids) - 1, st.session_state.index + 1)

# Fetch the current trial from database -- cached
@st.cache_data(hash_funcs={firestore.Client: id})
def get_document(doc_id):
    #current document
    doc_ref = db.collection('silver_trials').document(all_silver_ids[st.session_state.index])
    doc = doc_ref.get()
    return doc.id, doc.to_dict()

doc_id, doc_content = get_document(all_silver_ids[st.session_state.index])

# Display the current document
st.write(f"**Trial ID:** {doc_id}")
show_trial = st.toggle("Show Trial Details")
if show_trial:
    module.print_trial(doc_content)

#---------------- Step 2: Additional K-Shot Examples ----------------#

st.write("## Step 2: Additional K-Shot Examples")

col1, col2 = st.columns(2)

with col1:
    K = st.selectbox("Select K", options=[0, 1, 2, 3])

#k_examples = module.get_k_examples(K, st.session_state.current_name)

k_examples = module.few_shot_examples(K=K, seed=st.session_state.index, NCTId=doc_id)

st.divider()

#---------------- Step 3: System Instruction ----------------#

# Define the callback function
def update_prompt():
    # Get the current values of the text areas
    system_message = st.session_state.system

    prompt = module.get_final_prompt(K=K, seed=st.session_state.index, system_message=system_message, 
                            id=doc_id)

    return prompt

# Initialize the session state
if 'system' not in st.session_state:
    st.session_state.system = default_system 

# Text areas with the callback function
st.write("## Step 3: System Instruction")
st.text_area("System Instruction", value=st.session_state.system, height=250, 
                key='system', on_change=update_prompt)

if st.button("Update and Save"):
    #get the current text area value
    system_message = st.session_state.system

    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')

    #prepare the data to be saved 
    data = {
        'timestamp': firestore.SERVER_TIMESTAMP,
        'prompt': system_message,
        'model' : 'gpt-4-turbo'
    }

    #save the data
    doc_ref = db.collection('prompts').document('gpt-4-prompts').collection('all_prompts').document(timestamp_str)
    doc_ref.set(data)

    alert = st.success("System Instruction Updated and Saved Successfully!", icon="✅")
    time.sleep(3)
    alert.empty()

st.divider()

#---------------- Step 4: Final Prompt ----------------#

st.write("## Step 4: Final Prompt")

# Call the function once to display the initial prompt
prompt = update_prompt()

final_trial_info, final_baseline = module.row_to_info_converter(doc_content)

formatted_prompt = prompt.format(trial_info=final_trial_info)

st.write(formatted_prompt)

st.write(f"**Reference (collected using Clinicaltrials.gov API)**: {module.clean_string(final_baseline)}")

col1, col2, _, _, _ = st.columns(5)
with col1:
    run_gpt4 = st.button("Ask GPT-4 Turbo")
with col2:
    run_gpt3_5 = st.button("Ask GPT-3.5 Turbo")

if run_gpt4:
    model_name = "gpt-4-turbo"
    chat_llm = ChatOpenAI(model_name=model_name, temperature=0, model_kwargs={"seed": 1111}, n=3)
    with st.spinner("Generating..."):
        ret = chat_llm.invoke(formatted_prompt)
        time.sleep(1)

    st.write(ret.content)

if run_gpt3_5:
    model_name = "gpt-3.5-turbo"
    chat_llm = ChatOpenAI(model_name=model_name, temperature=0, model_kwargs={"seed": 1111})
    with st.spinner("Generating..."):
        ret = chat_llm.invoke(formatted_prompt)
        time.sleep(1)

    st.write(ret.content)


#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")
