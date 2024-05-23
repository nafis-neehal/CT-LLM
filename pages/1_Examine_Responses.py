import streamlit as st
import pandas as pd 
import module_light
# import module
# import os
# import time
# from datetime import date, datetime, timezone
# import itertools
# import defaults

# from langchain_openai import ChatOpenAI
# from langchain_community.llms import HuggingFaceEndpoint

# # from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain.prompts.prompt import PromptTemplate
# from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate

#from streamlit_option_menu import option_menu

from google.cloud import firestore

#---------------- Page Setup ----------------#
page_title = "Examine LLM Responses"
page_icon = "🔧"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Examine LLM Responses'*")

# #---------------- Sidebar ----------------#
# with st.sidebar:
#     st.write("## Settings")
#     oai_key = st.selectbox("Select OpenAI Key", ["RPI", "Personal"])
#     if oai_key == "Personal":
#         os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_personal"]
#     else:
#         os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_team"]
#     "---"

#---------------- Common Variables / Functions ----------------#
#db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

# Retrieve the Firebase credentials from Streamlit secrets
firebase_creds = st.secrets["firebase"]
db = module_light.load_firebase(firebase_creds)
id_ref = db.collection("All-IDs").document("Gold-100-ids")
id_dat = id_ref.get().to_dict()
all_gold_ids = id_dat['id_list']
#all_gold_ids = module.get_golddata_ids(sortit=True) 


#---------------- Main ----------------#
if 'index' not in st.session_state:
    st.session_state.index = 0

# Create next and previous buttons
c1, c2, _, _, _ = st.columns(5)
with c1:
    if st.button('◀ Previous'):
        # Decrement the index, ensuring it doesn't go below 0
        st.session_state.index = max(0, st.session_state.index - 1)
with c2:
    if st.button('Next ►'):
        # Increment the index, ensuring it doesn't go above the number of documents
        st.session_state.index = min(len(all_gold_ids) - 1, st.session_state.index + 1)

st.write(f"Trial {st.session_state.index + 1} of {len(all_gold_ids)}")
st.write(f"Trial ID: {all_gold_ids[st.session_state.index]}")

threeshot_example_ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']
if all_gold_ids[st.session_state.index] in threeshot_example_ids:
    st.write(f"🔴 :red[This is one of the three dummy trials used for testing and example purpose. \
             Please use the next or previous trial to view actual trial data.]")

else: 
    fetch_trial_button = st.button("Fetch Trial Data")
    if fetch_trial_button:
        doc_ref = db.collection("Gold-100").document(all_gold_ids[st.session_state.index])
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            with st.expander("Trial Information"):
                st.write(f"**Brief Title:** {data['BriefTitle']}")
                st.write(f"**Brief Summary:** {data['BriefSummary']}")
                st.write(f"**Eligibility Criteria:** \n\n {data['EligibilityCriteria']}")
                st.write(f"**Conditions:** {data['Conditions']}")
                st.write(f"**Interventions:** {data['Interventions']}")
                st.write(f"**Primary Outcomes:** {data['PrimaryOutcomes']}")
            with st.expander("Zero-shot Prompt"):
                module_light.build_zeroshot_prompt(data)
            with st.expander("Three-shot Prompt"):
                module_light.build_three_shot_prompt(data, db)
            with st.expander("Actual Responses"):
                st.write(f"**API Response:** {data['API_BaselineMeasures']}")
                st.write(f"**Publication Response:** {data['Paper_BaselineMeasures']}")
            with st.expander("LLM Generated Responses"):
                dat_ref = doc_ref.collection('gen-eval').get()
                for doc in dat_ref:
                    model_data = doc.to_dict()
                    st.write(f"**{doc.id}**")
                    st.write(f"{model_data['gen-response']}")
        else:
            st.write("No such document!")

#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")