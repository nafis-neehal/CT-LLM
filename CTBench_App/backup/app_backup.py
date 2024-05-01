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
page_title = "CTBench"
page_icon = "🧪" #"🤖" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'A Comprehensive Benchmark for Clinical Trial Design Automation Tasks'*")

#---------------- Sidebar ----------------#
with st.sidebar:
    st.write("## Site Wide Settings")
    oai_key = st.selectbox("Select OpenAI Key", ["Personal", "RPI"])
    if oai_key == "Personal":
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_personal"]
    else:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key_team"]
    "---"

#---------------- Navigation ----------------#
selected = option_menu(
    menu_title = None, 
    options = ["Prompt Engineering", "GPT-4 as Evaluator", "Human as Evaluator"],
    icons = ["gear", "robot", "person-arms-up"],
    orientation = "horizontal",
)

#---------------- Common Variables / Functions ----------------#
db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")


###################################################
#                                                 # 
#               Prompt Engineering                #
#                                                 #
###################################################

if selected == "Prompt Engineering":

    #data
    df = module.get_silverdata_full()

    # Initialize a session state variable to keep track of the current name
    if 'current_name' not in st.session_state:
        st.session_state.current_name = df.iloc[0]['NCTId']  # Start with the first name

    # Default values for the system and human instructions

    docs = db.collection('prompts').document('gpt-4-prompts').collection('all_prompts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
    for doc in docs:
        default_system = doc.to_dict()['prompt']

    #---------------- Step 1: Current Query Trial ----------------#

    st. write("# Designing Final Prompt")

    st.write("## Step 1: Current Query Trial")
    
    # Get the current index based on the name
    current_index = df.index[df['NCTId'] == st.session_state.current_name][0]

    # Function to decrement the index to see the previous row
    def previous():
        current_index = df.index[df['NCTId'] == st.session_state.current_name][0]
        if current_index > 0:
            st.session_state.current_name = df.iloc[current_index - 1]['NCTId']

    # Function to increment the index to see the next row
    def next():
        current_index = df.index[df['NCTId'] == st.session_state.current_name][0]
        if current_index < len(df) - 1:
            st.session_state.current_name = df.iloc[current_index + 1]['NCTId']

    # Buttons for navigation
    col1, col2, _, _ = st.columns(4)
    with col1:
        st.button("◀ Previous Trial", on_click=previous)
    with col2:
        st.button("Next Trial ►", on_click=next)

    st.write(f"Trial {current_index + 1} of {len(df)}")



    toggle = st.toggle("Show Trial Details")

    if toggle:
        module.print_trial(df.loc[current_index])

    else:
        st.write(df.loc[current_index])

    st.divider()

    #---------------- Step 2: Additional K-Shot Examples ----------------#

    st.write("## Step 2: Additional K-Shot Examples")

    col1, col2 = st.columns(2)

    with col1:
        K = st.selectbox("Select K", options=[0, 1, 2, 3])

    #k_examples = module.get_k_examples(K, st.session_state.current_name)
    k_examples = module.few_shot_examples(K=K, seed=current_index, NCTId=st.session_state.current_name)

    # for i, example in enumerate(k_examples):
    #     st.write(f"#### Example {i + 1}")
    #     st.write(example)

    st.divider()

    #---------------- Step 3: System Instruction ----------------#

    # Define the callback function
    def update_prompt():
        # Get the current values of the text areas
        system_message = st.session_state.system

        prompt = module.get_final_prompt(K=K, seed=current_index, system_message=system_message, 
                                id=st.session_state.current_name)

        return prompt

    # Initialize the session state
    if 'system' not in st.session_state:
        st.session_state.system = default_system 

    # Text areas with the callback function
    st.write("## Step 3: System Instruction")
    st.text_area("System Instruction", value=st.session_state.system, height=250, 
                 key='system', on_change=update_prompt)
    
    st.write(date.today())
    
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



    st.divider()

    #---------------- Step 4: Final Prompt ----------------#

    st.write("## Step 4: Final Prompt")

    # Call the function once to display the initial prompt
    prompt = update_prompt()

    final_trial_info, final_baseline = module.row_to_info_converter(df.loc[current_index])

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




###################################################
#                                                 # 
#               GPT-4 as Evaluator                #
#                                                 #
###################################################
if selected == "GPT-4 as Evaluator":
    st.write("# GPT-4 as Evaluator")

    #------------ Step 1: LLM Selection ------------#
    st.write("## Step 1: Select LLM") 
    llm = st.selectbox("Select LLM whose response you want to evaluate", 
                       options=["GPT-4-Turbo ✅", "GPT-3.5-Turbo ❌", "Mixtral ❌"])

    #------------ Step 2: Select Trial ------------#
    st.write("## Step 2: Select Trial")

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

    st.write(f"## Step 3: Evaluate {llm[:-1]} responses by GPT-4-Turbo")

    baseline = module.clean_string(df_100.loc[current_index_100]['BaselineMeasures'])
    zeroshot = module.clean_string(df_100.loc[current_index_100]['ZeroShot'])
    oneshot = module.clean_string(df_100.loc[current_index_100]['OneShot'])
    twoshot = module.clean_string(df_100.loc[current_index_100]['TwoShot'])
    threeshot = module.clean_string(df_100.loc[current_index_100]['ThreeShot'])


    with st.expander(f"Zero-Shot by {llm}"):
        st.write(f"**Baseline Response**")
        st.write(baseline)
        st.write(f"**Zero-Shot Response**")
        st.write(zeroshot)

        system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                 module.get_list_from_string(zeroshot))
        
        st.divider()

        t0 = st.toggle("See Prompt", key="t0")
        if t0:
            st.write(system + "\n\n" + prompt)

        st.divider()

        run_eval = st.button("Evaluate with GPT-4", key="b0")
        if run_eval:
            st.json(module.get_gpt4_eval_score(system, prompt))
    
    with st.expander(f"One-Shot by {llm}"):
        st.write(f"**Baseline Response**")
        st.write(baseline)
        st.write(f"**One-Shot Response**")
        st.write(oneshot)

        system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                 module.get_list_from_string(oneshot))
        
        st.divider()

        t1 = st.toggle("See Prompt", key = "t1")
        if t1:
            st.write(system + "\n\n" + prompt)

        st.divider()

        run_eval = st.button("Evaluate with GPT-4", key="b1")
        if run_eval:
            st.json(module.get_gpt4_eval_score(system, prompt))

    with st.expander(f"Two-Shot by {llm}"):
        st.write(f"**Baseline Response**")
        st.write(baseline)
        st.write(f"**Two-Shot Response**")
        st.write(twoshot)

        system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                 module.get_list_from_string(twoshot))
        
        st.divider()

        t2 = st.toggle("See Prompt", key ="t2")
        if t2:
            st.write(system + "\n\n" + prompt)

        st.divider()

        run_eval = st.button("Evaluate with GPT-4", key="b2")
        if run_eval:
            st.json(module.get_gpt4_eval_score(system, prompt))
    
    with st.expander(f"Three-Shot by {llm}"):
        st.write(f"**Baseline Response**")
        st.write(baseline)
        st.write(f"**Three-Shot Response**")
        st.write(threeshot)

        system, prompt = module.get_gpt4_eval_prompt(module.get_list_from_string(baseline), 
                                                 module.get_list_from_string(threeshot))
        
        st.divider()

        t3 = st.toggle("See Prompt", key = "t3")
        if t3:
            st.write(system + "\n\n" + prompt)

        st.divider()

        run_eval = st.button("Evaluate with GPT-4", key="b3")
        if run_eval:
            st.json(module.get_gpt4_eval_score(system, prompt))
            

###################################################
#                                                 #
#               Human as Evaluator                #
#                                                 #
###################################################

if selected == "Human as Evaluator":
    st.write("# Human as Evaluator")
    st.write("Coming Soon...")

    # toggle = st.toggle("DB Connect")
    # if toggle:


    #     #db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

    #     all_silver_ids = module.get_silverdata_ids()

    #     # Initialize a session state for the current index
    #     if 'index' not in st.session_state:
    #         st.session_state.index = 0

    #     st.write(f"Total number of silver trials: {len(all_silver_ids)}")
    #     st.write(f"Current index: {st.session_state.index}")

    #     # Create next and previous buttons
    #     if st.button('Previous'):
    #         # Decrement the index, ensuring it doesn't go below 0
    #         st.session_state.index = max(0, st.session_state.index - 1)

    #     if st.button('Next'):
    #         # Increment the index, ensuring it doesn't go above the number of documents
    #         st.session_state.index = min(len(all_silver_ids) - 1, st.session_state.index + 1)

    #     # Fetch the document
    #     @st.cache_data(hash_funcs={firestore.Client: id})
    #     def get_document(doc_id):
    #         #current document
    #         doc_ref = db.collection('silver_trials').document(all_silver_ids[st.session_state.index])
    #         doc = doc_ref.get()
    #         return doc.id, doc.to_dict()
        
    #     doc_id, doc_content = get_document(all_silver_ids[st.session_state.index])

    #     st.write(firestore.SERVER_TIMESTAMP)
        
    #     # Display the current document
    #     st.write(f"**Trial ID:** {doc_id}")
    #     for key, value in doc_content.items():
    #         st.write(f"**{key}** : {value}")

    


#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")


