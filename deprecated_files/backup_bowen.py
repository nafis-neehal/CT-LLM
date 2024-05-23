import streamlit as st
import json
from google.cloud import firestore
#import module
import module_light
import pandas as pd
# import os
# import time
#from datetime import date, datetime, timezone
# import itertools
# import defaults

#---------------- Page Setup ----------------#
page_title = "Human Evaluation by Bowen"
page_icon = "👨‍💼"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Human-in-the-loop evaluation'*")

#---------------- Common Variables / Functions ----------------#
db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

id_ref = db.collection("All-IDs").document("Bowen-Gold-100-ids")
id_dat = id_ref.get().to_dict()
all_gold_ids = id_dat['id_list']
if 'last_saved_id' in id_dat:
    last_saved_id_index = all_gold_ids.index(id_dat['last_saved_id'])
else:
    last_saved_id_index = -1

#all_gold_ids = module.get_golddata_ids(sortit=True)

# Function to fetch trial data and update session state variables
def fetch_trial_data(index, gen_model):
    doc_ref = db.collection("Gold-100").document(all_gold_ids[index])
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        reference_list = module_light.extract_elements(data['Paper_BaselineMeasures'])
        #dat_ref = doc_ref.collection('gen-eval').document('gpt4-omni-ts')
        dat_ref = doc_ref.collection('gen-eval').document(gen_model)
        dat = dat_ref.get()
        model_data = dat.to_dict()
        candidate_list = module_light.extract_elements(model_data['gen-response'])
        st.session_state.trial_data = data
        st.session_state.reference_list = reference_list.copy()
        st.session_state.candidate_list = candidate_list.copy()
        st.session_state.reference_selection = None
        st.session_state.candidate_selection = None
        st.session_state.matched_pairs = []
        st.session_state.finished = False
        st.session_state.additional_relevant_candidate_features = []

#---------------- Main ----------------#
if 'index' not in st.session_state:
    #st.session_state.index = 0
    st.session_state.index = last_saved_id_index + 1

# select_generation_model = st.selectbox("Select Generation Model", ["gpt4-omni-zs", "gpt4-omni-ts", 
#                                                      "gpt4-turbo-zs", "gpt4-turbo-ts", 
#                                                      "llama3-70b-in-zs", "llama3-70b-in-ts"])

select_generation_model = st.selectbox("Select Generation Model", ["gpt4-omni-ts"])

# Create next and previous buttons
c1, c2, _, _, _ = st.columns(5)
with c1:
    if st.button('◀ Previous'):
        # Decrement the index, ensuring it doesn't go below 0
        st.session_state.index = max(0, st.session_state.index - 1)
        fetch_trial_data(st.session_state.index, select_generation_model)
with c2:
    if st.button('Next ►'):
        # Increment the index, ensuring it doesn't go above the number of documents
        st.session_state.index = min(len(all_gold_ids) - 1, st.session_state.index + 1)
        fetch_trial_data(st.session_state.index, select_generation_model)

if st.session_state.index <= last_saved_id_index:
    st.warning(f"You are going back to a previously saved trial with trial id {all_gold_ids[st.session_state.index]}. Please review the response before submitting.")


st.write(f"Trial {st.session_state.index + 1} of {len(all_gold_ids)}")
st.write(f"Trial ID: {all_gold_ids[st.session_state.index]}")

threeshot_example_ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']
if all_gold_ids[st.session_state.index] in threeshot_example_ids:
    st.write(f"🔴 :red[This is one of the three dummy trials used for testing and example purposes. \
             Please use the next or previous trial to view actual trial data.]")
else:
    if st.button("Fetch Trial Data"):
        fetch_trial_data(st.session_state.index, select_generation_model)

    if 'reference_list' in st.session_state and 'candidate_list' in st.session_state:

        #print trial
        trial_data = st.session_state.trial_data
        with st.expander("Trial Information"):
                st.write(f"**Brief Title:** {trial_data['BriefTitle']}")
                st.write(f"**Brief Summary:** {trial_data['BriefSummary']}")
                st.write(f"**Eligibility Criteria:** \n\n {trial_data['EligibilityCriteria']}")
                st.write(f"**Conditions:** {trial_data['Conditions']}")
                st.write(f"**Interventions:** {trial_data['Interventions']}")
                st.write(f"**Primary Outcomes:** {trial_data['PrimaryOutcomes']}")


        col1, col2 = st.columns(2)

        with col1:
            # Display reference list with radio buttons for selection
            st.subheader("Select an item from the Reference List")
            st.session_state.reference_selection = st.radio(
                "Reference List",
                st.session_state.reference_list,
                index=st.session_state.reference_list.index(st.session_state.reference_selection) if st.session_state.reference_selection in st.session_state.reference_list else 0
            )

        with col2:
            # Display candidate list with radio buttons for selection
            st.subheader("Select an item from the Candidate List")
            st.session_state.candidate_selection = st.radio(
                "Candidate List",
                st.session_state.candidate_list,
                index=st.session_state.candidate_list.index(st.session_state.candidate_selection) if st.session_state.candidate_selection in st.session_state.candidate_list else 0
            )

        col1, col2, col3 = st.columns(3)
        # Match button
        with col1:
            if st.button("Match Selected Items"):
                if st.session_state.reference_selection and st.session_state.candidate_selection:
                    st.session_state.matched_pairs.append((st.session_state.reference_selection, st.session_state.candidate_selection))
                    st.session_state.reference_list.remove(st.session_state.reference_selection)
                    st.session_state.candidate_list.remove(st.session_state.candidate_selection)
                    st.session_state.reference_selection = None
                    st.session_state.candidate_selection = None
                    st.success("Items matched and removed from lists.")
                else:
                    st.error("Please select an item from both lists to match.")

        # Clear selections button
        with col2:
            if st.button("Clear Selections"):
                st.session_state.reference_selection = None
                st.session_state.candidate_selection = None
                st.success("Selections cleared.")

        # Reset lists button
        with col3:
            if st.button("Reset Lists"):
                fetch_trial_data(st.session_state.index, select_generation_model)

    st.subheader("Matched Pairs")
    if 'matched_pairs' in st.session_state and st.session_state.matched_pairs:
        for pair in st.session_state.matched_pairs:
            st.write(f"{pair[0]} - {pair[1]}")
    else:
        st.write("No matched pairs yet.")

    st.subheader("Unmatched Items")
    if 'reference_list' in st.session_state:
        current_reference_list = st.session_state.reference_list
    else:
        current_reference_list = []
        #st.write("reference List:", st.session_state.reference_list)
    if 'candidate_list' in st.session_state:
        current_candidate_list = st.session_state.candidate_list
    else:
        current_candidate_list = []
        #st.write("Candidate List:", st.session_state.candidate_list)
    unmatched_dict = {"unmatched_reference": current_reference_list, "unmatched_candidate": current_candidate_list}
    unmatched_df = pd.DataFrame([unmatched_dict])
    html = unmatched_df.to_html(index=False).replace('<table', '<table style="font-size:8px"')
    st.markdown(html, unsafe_allow_html=True)
    st.write(" ")

    # Finish Matching button
    if st.button("Finish Matching"):
        st.session_state.finished = True

    # Display checklist for additional relevant candidate features
    if 'finished' in st.session_state and st.session_state.finished:
        st.subheader("Select Additional Relevant Candidate Features")

        # Initialize the selected items in st.session_state if they're not already initialized
        if 'additional_relevant_candidate_features' not in st.session_state:
            st.session_state.additional_relevant_candidate_features = []
        
        # Use the selected items in st.session_state as the default value for the st.multiselect widget
        additional_relevant_candidate_features = st.multiselect(
            "Remaining Candidate Features",
            st.session_state.candidate_list,
            default=st.session_state.additional_relevant_candidate_features
        )

        # Update the session state with the selected features
        st.session_state.additional_relevant_candidate_features = additional_relevant_candidate_features

        # Update the remaining candidate features
        remaining_candidate_features = [feature for feature in st.session_state.candidate_list if feature not in st.session_state.additional_relevant_candidate_features]

        if st.button("Review Final Response"):
            result = {
                "matched_features": st.session_state.matched_pairs,
                "remaining_reference_features": st.session_state.reference_list,
                "remaining_candidate_features": remaining_candidate_features,
                "additional_relevant_candidate_features": st.session_state.additional_relevant_candidate_features
            }
            matches_df = pd.DataFrame([result])

            st.session_state.result = result
            st.subheader("Final Response")
            #st.json(st.session_state.result)

            # Convert the DataFrame to HTML and add some CSS to change the font size
            html = matches_df.to_html(index=False).replace('<table', '<table style="font-size:8px"')

            # Display the HTML in the Streamlit app
            st.markdown(html, unsafe_allow_html=True)

            st.write(" ")
            
        # Save response button
        if st.button("Submit"):
            if 'result' in st.session_state:
                doc_ref = db.collection('Gold-100').document(all_gold_ids[st.session_state.index]).collection('gen-eval').document(select_generation_model)
                doc_ref.set({
                    "human-response": json.dumps(st.session_state.result),
                    "human-name": "Bowen"
                }, merge=True)

                if st.session_state.index > last_saved_id_index:
                    doc_ref = db.collection('All-IDs').document('Bowen-Gold-100-ids')
                    doc_ref.set({
                        "last_saved_id": all_gold_ids[st.session_state.index]
                    }, merge=True)
                    st.success(f"Response saved!!")
                else:
                    st.success(f"Response updated!!")
            else:
                st.error("No result to save.")
