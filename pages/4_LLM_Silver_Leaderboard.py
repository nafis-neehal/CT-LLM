import streamlit as st
import pandas as pd 
import module_lite
import json 

from google.cloud import firestore

#---------------- Page Setup ----------------#
page_title = "LLM Silver Leaderboard"
page_icon = "📊"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'LLM Silver Leaderboard'*")

#---------------- Common Variables / Functions ----------------#
#db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

# Retrieve the Firebase credentials from Streamlit secrets
firebase_creds = st.secrets["firebase"]
db = module_lite.load_firebase(firebase_creds)
id_ref = db.collection("All-IDs").document("Silver-Data-ids")
id_dat = id_ref.get().to_dict()
all_silver_ids = id_dat['id_list']

#---------------- Main ----------------#

# """
# DataFrame structure
# Trial ID - Generation Model - Evaluation Model - Precision - Recall - F1 
# """

score = pd.DataFrame(columns=['Trial_ID', 'Generation_Model', 'Evaluation_Model', 'Precision', 'Recall', 'F1'])

run_button = st.button("Re-Run Leaderboard")

show_current_leaderboard = st.button("Show Latest Leaderboard")

if run_button:

    progress_text = "Calulating... Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    avoid_ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']

    #for each trial in the Silver-Data collection
    for index, id in enumerate(all_silver_ids):

        #my_bar.progress(index+1, text=progress_text)
        my_bar.progress(int(((index+1)/len(all_silver_ids)) * 100), text=progress_text)

        trial_id = id

        if trial_id in avoid_ids:
            continue

        # ##################
        # if trial_id < 'NCT02960204': ###this is where gpt4-omni-match generation ended (660 trials)
        #     continue
        # ##################

        gen_ref = db.collection("Silver-Data").document(trial_id).collection("gen-eval").get()

        #for each model we tried for generation - 4 models
        for gen in gen_ref:

            gen_model_name = gen.id
            #st.write(gen_model_name)
            
            gen_data = gen.to_dict()

            #bert-score-06
            bs06_prec = gen_data['bert-scores-06']['precision']
            bs06_rec = gen_data['bert-scores-06']['recall']
            bs06_f1 = gen_data['bert-scores-06']['f1']
            new_row = pd.DataFrame({'Trial_ID': [trial_id], 'Generation_Model': [gen_model_name], 
                                'Evaluation_Model': ['bert-scores-06'], 
                                'Precision': [bs06_prec], 'Recall': [bs06_rec], 'F1': [bs06_f1]})
            score = pd.concat([score, new_row], ignore_index=True)

            #bert-score-07
            bs07_prec = gen_data['bert-scores-07']['precision']
            bs07_rec = gen_data['bert-scores-07']['recall']
            bs07_f1 = gen_data['bert-scores-07']['f1']
            new_row = pd.DataFrame({'Trial_ID': [trial_id], 'Generation_Model': [gen_model_name], 
                                'Evaluation_Model': ['bert-scores-07'], 
                                'Precision': [bs07_prec], 'Recall': [bs07_rec], 'F1': [bs07_f1]})
            score = pd.concat([score, new_row], ignore_index=True)
            
            #bert-score-08
            bs08_prec = gen_data['bert-scores-08']['precision']
            bs08_rec = gen_data['bert-scores-08']['recall']
            bs08_f1 = gen_data['bert-scores-08']['f1']
            new_row = pd.DataFrame({'Trial_ID': [trial_id], 'Generation_Model': [gen_model_name], 
                                'Evaluation_Model': ['bert-scores-08'], 
                                'Precision': [bs08_prec], 'Recall': [bs08_rec], 'F1': [bs08_f1]})
            score = pd.concat([score, new_row], ignore_index=True)
            
            #bert-score-09
            bs09_prec = gen_data['bert-scores-09']['precision']
            bs09_rec = gen_data['bert-scores-09']['recall']
            bs09_f1 = gen_data['bert-scores-09']['f1']
            new_row = pd.DataFrame({'Trial_ID': [trial_id], 'Generation_Model': [gen_model_name], 
                                'Evaluation_Model': ['bert-scores-09'], 
                                'Precision': [bs09_prec], 'Recall': [bs09_rec], 'F1': [bs09_f1]})
            
            score = pd.concat([score, new_row], ignore_index=True)

            #gpt4-omni-score
            gpt4_omni_matches = json.loads(gen_data['gpt4-omni-matches'])
            gpt4_omni_scores = module_lite.match_to_score(gpt4_omni_matches)

            new_row = pd.DataFrame({'Trial_ID': [trial_id], 'Generation_Model': [gen_model_name], 
                                'Evaluation_Model': ['gpt4-omni-score'], 
                                'Precision': [gpt4_omni_scores['precision']], 
                                'Recall': [gpt4_omni_scores['recall']], 
                                'F1': [gpt4_omni_scores['f1']]})
            
            score = pd.concat([score, new_row], ignore_index=True)

    my_bar.empty()

    df = score.drop(['Trial_ID'], axis=1)

    # Calculate mean and standard deviation
    mean_df = df.groupby(['Generation_Model', 'Evaluation_Model']).mean().reset_index()
    std_df = df.groupby(['Generation_Model', 'Evaluation_Model']).std().reset_index()

    # Merge mean and standard deviation DataFrames
    aggregate_score = pd.merge(mean_df, std_df, 
                               on=['Generation_Model', 'Evaluation_Model'], 
                               suffixes=('_mean', '_std'))
    
    st.table(aggregate_score)

    #save to database
    for index, row in aggregate_score.iterrows():
        doc_ref = db.collection("silver-leaderboard-scores").document(row['Generation_Model']).collection(row['Evaluation_Model']).document('scores')
        doc_ref.set({
            'Precision_mean': row['Precision_mean'],
            'Precision_std': row['Precision_std'],
            'Recall_mean': row['Recall_mean'],
            'Recall_std': row['Recall_std'],
            'F1_mean': row['F1_mean'],
            'F1_std': row['F1_std']
        }, merge=True)

    st.success("Leaderboard updated successfully!")

    score.to_csv("silver_leaderboard.csv", index=False)
    aggregate_score.to_csv("silver_aggregate_leaderboard.csv", index=False)


# # elif show_current_leaderboard:
if show_current_leaderboard:

    st.header("Current Leaderboard on Silver Dataset")

    aggregate_score = pd.DataFrame(columns=['Generation_Model', 'Evaluation_Model', 'Precision_mean', 
                                  'Precision_std', 'Recall_mean', 'Recall_std', 'F1_mean', 'F1_std'])

    docs = db.collection("silver-leaderboard-scores").list_documents()

    my_bar = st.progress(0, text="Generating Leaderboard...")

    c = 0
    for doc in docs:
        doc_name = doc.id
        doc_ref = db.collection("silver-leaderboard-scores").document(doc_name).collections()
        for sub_coll in doc_ref:
            my_bar.progress(c+1, text="Generating Leaderboard...")
            sub_coll_name = sub_coll.id
            if sub_coll_name == 'gpt4-omni-score-no_context':
                continue
            sub_coll_ref = db.collection("silver-leaderboard-scores").document(doc_name).collection(sub_coll_name).document('scores')
            sub_coll_dat = sub_coll_ref.get().to_dict()

            new_row = pd.DataFrame({'Generation_Model': [doc_name], 'Evaluation_Model': [sub_coll_name], 
                                'Precision_mean': [sub_coll_dat['Precision_mean']], 'Precision_std': [sub_coll_dat['Precision_std']], 
                                'Recall_mean': [sub_coll_dat['Recall_mean']], 'Recall_std': [sub_coll_dat['Recall_std']], 
                                'F1_mean': [sub_coll_dat['F1_mean']], 'F1_std': [sub_coll_dat['F1_std']]})

            aggregate_score = pd.concat([aggregate_score, new_row], ignore_index=True)

            c+=1

    my_bar.empty()

    #st.write(aggregate_score)

    #Plotting Precision, Recall, and F1
    fig_precision = module_lite.plot_metrics(aggregate_score.copy(), 'Precision', 'CT-Repo', save_path="silver_avg_precision.png")
    fig_recall = module_lite.plot_metrics(aggregate_score.copy(), 'Recall', 'CT-Repo', save_path="silver_avg_recall.png")
    fig_f1 = module_lite.plot_metrics(aggregate_score.copy(), 'F1', 'CT-Repo', save_path="silver_avg_f1.png")

    # fig_precision = module_lite.plot_metrics(aggregate_score.copy(), 'Precision', 'CT-Repo')
    # fig_recall = module_lite.plot_metrics(aggregate_score.copy(), 'Recall', 'CT-Repo')
    # fig_f1 = module_lite.plot_metrics(aggregate_score.copy(), 'F1', 'CT-Repo')

    st.plotly_chart(fig_precision)
    st.plotly_chart(fig_recall)
    st.plotly_chart(fig_f1)


#---------------- Footer ----------------#
st.caption("© 2024-2025 CTBench. All Rights Reserved.")
st.caption("Developed by [Nafis Neehal](https://nafis-neehal.github.io/) in collaboration with RPI IDEA and IBM")