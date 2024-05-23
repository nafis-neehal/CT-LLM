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
page_title = "LLM as Evaluator"
page_icon = "🤖"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'LLM as Evaluator'*")

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
#               GPT-4 as Evaluator                #
#                                                 #
###################################################


#------------ Step 1: LLM Selection ------------#
st.write("## Step 1: Select LLM") 
llm = st.selectbox("Select LLM to use as evaluator:", 
                    options=["GPT-4-Turbo", "GPT-3.5-Turbo", "BERT"])

#------------ Step 2: Select Trial ------------#
st.write("## Step 2: Select Trial")
st.write("We are looking at a subset of ~100 trials from the CTBench Silver Dataset here. Responses for the remaining trials will be generated after prompt is finalized.")

df = module.get_silverdata_full()
df_100 = module.get_silverdata_100()

# Initialize a session state variable to keep track of the current name
if 'current_name_100' not in st.session_state:
    st.session_state.current_name_100 = df_100.iloc[0]['TrialID']  # Start with the first name

# Get the current index based on the name
current_index_100 = df_100.index[df_100['TrialID'] == st.session_state.current_name_100][0]

#show current trial information
current_id = df_100.loc[current_index_100]['TrialID']
current_trial_info = df.index[df['NCTId'] == current_id][0]

#st.write(pd.concat([df.loc[current_trial_info], df_100.loc[current_index_100][2:]]))

# Function to decrement the index to see the previous row
def previous():
    current_index_100 = df_100.index[df_100['TrialID'] == st.session_state.current_name_100][0]
    if current_index_100 > 0:
        st.session_state.current_name_100 = df_100.iloc[current_index_100 - 1]['TrialID']

# Function to increment the index to see the next row
def next():
    current_index_100 = df_100.index[df_100['TrialID'] == st.session_state.current_name_100][0]
    if current_index_100 < len(df_100) - 1:
        st.session_state.current_name_100 = df_100.iloc[current_index_100 + 1]['TrialID']

# Buttons for navigation
col1, col2, _, _ = st.columns(4)
with col1:
    st.button("◀ Previous Trial", on_click=previous)
with col2:
    st.button("Next Trial ►", on_click=next)

st.write(f"Trial {current_index_100 + 1} of {len(df_100)}")

toggle_100 = st.toggle("Show Trial Details")
if toggle_100:
    # module.print_trial(pd.concat([df.loc[current_trial_info], df_100.loc[current_index_100][2:]]),
    #                 print_responses=True)
    module.print_trial(df.loc[current_trial_info])
else:
    #st.write(pd.concat([df.loc[current_trial_info], df_100.loc[current_index_100][2:]]))
    st.dataframe(df.loc[current_trial_info], use_container_width=True)

#------------ Step 3: Evaluate Responses ------------#

st.write(f"## Step 3: Evaluate with {llm}")

baseline = module.clean_string(df_100.loc[current_index_100]['BaselineMeasures'])
zeroshot = module.clean_string(df_100.loc[current_index_100]['ZeroShot'])
oneshot = module.clean_string(df_100.loc[current_index_100]['OneShot'])
twoshot = module.clean_string(df_100.loc[current_index_100]['TwoShot'])
threeshot = module.clean_string(df_100.loc[current_index_100]['ThreeShot'])


with st.expander(f"Zero-Shot"):
    st.write(f"**Baseline Response**")
    st.write(baseline)
    st.write(f"**Zero-Shot Response**")
    st.write(zeroshot)

    system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                module.get_list_from_string(zeroshot))
    
    st.divider()
    if llm == "GPT-4-Turbo" or llm == "GPT-3.5-Turbo":
        t0 = st.toggle("See Prompt", key="t0")
        if t0:
            st.write(system + "\n\n" + prompt)

        st.divider()

    run_eval = st.button("Evaluate", key="b0")
    if run_eval:
        if llm == "GPT-4-Turbo": st.json(module.get_gpt4_eval_score(system, prompt))
        elif llm == "GPT-3.5-Turbo": st.json(module.get_gpt35_eval_score(system, prompt))
        else: module.get_bert_eval_score2(baseline, zeroshot, 0.5)

with st.expander(f"One-Shot"):
    st.write(f"**Baseline Response**")
    st.write(baseline)
    st.write(f"**One-Shot Response**")
    st.write(oneshot)

    system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                module.get_list_from_string(oneshot))
    
    st.divider()
    if llm == "GPT-4-Turbo" or llm == "GPT-3.5-Turbo":
        t1 = st.toggle("See Prompt", key = "t1")
        if t1:
            st.write(system + "\n\n" + prompt)

        st.divider()

    run_eval = st.button("Evaluate", key="b1")
    if run_eval:
        #st.json(module.get_gpt4_eval_score(system, prompt))
        if llm == "GPT-4-Turbo": st.json(module.get_gpt4_eval_score(system, prompt))
        elif llm == "GPT-3.5-Turbo": st.json(module.get_gpt35_eval_score(system, prompt))
        else: module.get_bert_eval_score(baseline, oneshot)

with st.expander(f"Two-Shot"):
    st.write(f"**Baseline Response**")
    st.write(baseline)
    st.write(f"**Two-Shot Response**")
    st.write(twoshot)

    system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                module.get_list_from_string(twoshot))
    
    st.divider()

    if llm == "GPT-4-Turbo" or llm == "GPT-3.5-Turbo":
        t2 = st.toggle("See Prompt", key ="t2")
        if t2:
            st.write(system + "\n\n" + prompt)

        st.divider()

    run_eval = st.button("Evaluate", key="b2")
    if run_eval:
        #st.json(module.get_gpt4_eval_score(system, prompt))
        if llm == "GPT-4-Turbo": st.json(module.get_gpt4_eval_score(system, prompt))
        elif llm == "GPT-3.5-Turbo": st.json(module.get_gpt35_eval_score(system, prompt))
        else: module.get_bert_eval_score(baseline, twoshot)

with st.expander(f"Three-Shot"):
    st.write(f"**Baseline Response**")
    st.write(baseline)
    st.write(f"**Three-Shot Response**")
    st.write(threeshot)

    system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                module.get_list_from_string(threeshot))
    
    st.divider()

    if llm == "GPT-4-Turbo" or llm == "GPT-3.5-Turbo":
        t3 = st.toggle("See Prompt", key = "t3")
        if t3:
            st.write(system + "\n\n" + prompt)

        st.divider()

    run_eval = st.button("Evaluate", key="b3")
    if run_eval:
        #st.json(module.get_gpt4_eval_score(system, prompt))
        if llm == "GPT-4-Turbo": st.json(module.get_gpt4_eval_score(system, prompt))
        elif llm == "GPT-3.5-Turbo": st.json(module.get_gpt35_eval_score(system, prompt))
        else: module.get_bert_eval_score(baseline, threeshot)



#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")