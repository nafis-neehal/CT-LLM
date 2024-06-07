import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd 
import module_lite
import json 

from google.cloud import firestore

#---------------- Page Setup ----------------#
page_title = "Combined Leaderboard"
page_icon = "📊"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")
st.title(page_title + " " + page_icon)
st.write("*'Combined Leaderboard'*")

#---------------- Common Variables / Functions ----------------#
#db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

# Retrieve the Firebase credentials from Streamlit secrets
firebase_creds = st.secrets["firebase"]
db = module_lite.load_firebase(firebase_creds)

########### Gold 100 ###########

run_button = st.button("Re-Run Leaderboard")

if run_button:

    st.header("Current Leaderboard on Gold-100 Dataset")

    aggregate_score = pd.DataFrame(columns=['Generation_Model', 'Evaluation_Model', 'Precision_mean', 
                                    'Precision_std', 'Recall_mean', 'Recall_std', 'F1_mean', 'F1_std'])

    docs = db.collection("leaderboard-scores").list_documents()

    my_bar = st.progress(0, text="Generating Leaderboard...")

    c = 0
    for doc in docs:
        doc_name = doc.id
        if doc_name == 'gpt4-turbo-zs' or doc_name == 'gpt4-turbo-ts':
            continue
        doc_ref = db.collection("leaderboard-scores").document(doc_name).collections()
        for sub_coll in doc_ref:
            my_bar.progress(c+1, text="Generating Leaderboard...")
            sub_coll_name = sub_coll.id
            if sub_coll_name not in ['gpt4-omni-score', 'bert-scores-07']:
                continue
            sub_coll_ref = db.collection("leaderboard-scores").document(doc_name).collection(sub_coll_name).document('scores')
            sub_coll_dat = sub_coll_ref.get().to_dict()

            new_row = pd.DataFrame({'Generation_Model': [doc_name], 'Evaluation_Model': [sub_coll_name], 
                                'Precision_mean': [sub_coll_dat['Precision_mean']], 'Precision_std': [sub_coll_dat['Precision_std']], 
                                'Recall_mean': [sub_coll_dat['Recall_mean']], 'Recall_std': [sub_coll_dat['Recall_std']], 
                                'F1_mean': [sub_coll_dat['F1_mean']], 'F1_std': [sub_coll_dat['F1_std']]})

            aggregate_score = pd.concat([aggregate_score, new_row], ignore_index=True)

            c+=1

    my_bar.empty()

    #Plotting Precision, Recall, and F1
    fig_precision_gold = module_lite.plot_metrics(aggregate_score.copy(), 'Precision', 'CT-Pub')
    fig_recall_gold = module_lite.plot_metrics(aggregate_score.copy(), 'Recall', 'CT-Pub')
    fig_f1_gold = module_lite.plot_metrics(aggregate_score.copy(), 'F1', 'CT-Pub')

    ########### Silver ###########

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
            if sub_coll_name not in ['gpt4-omni-score', 'bert-scores-07']:
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

    #Plotting Precision, Recall, and F1
    fig_precision_silver = module_lite.plot_metrics(aggregate_score.copy(), 'Precision', 'CT-Repo')
    fig_recall_silver = module_lite.plot_metrics(aggregate_score.copy(), 'Recall', 'CT-Repo')
    fig_f1_silver = module_lite.plot_metrics(aggregate_score.copy(), 'F1', 'CT-Repo')

    ##############################

    # Create a 3x2 subplot layout
    fig_combined = make_subplots(rows=3, cols=2, 
                                 subplot_titles=('(a) Mean Recall in CT-Pub', '(b) Mean Recall in CT-Repo', 
                                                 '(c) Mean Precision in CT-Pub', '(d) Mean Precision in CT-Repo',
                                                 '(e) Mean F1 in CT-Pub', '(f) Mean F1 in CT-Repo'),
                                                 vertical_spacing=0.24)  # Adjust this value to control the vertical spacing)

    def add_traces_from_fig(fig, row, col, show_legend_flags, metric_name):
        for trace in fig['data']:
            show_legend = show_legend_flags.get(trace.name, True)
            trace.showlegend = show_legend
            show_legend_flags[trace.name] = False
            fig_combined.add_trace(trace, row=row, col=col)
        fig_combined.update_yaxes(title_text=f"Mean {metric_name}", row=row, col=col, range=[0, 0.75])

    show_legend_flags = {}

    add_traces_from_fig(fig_recall_gold, 1, 1, show_legend_flags, "Recall")
    add_traces_from_fig(fig_recall_silver, 1, 2, show_legend_flags, "Recall")
    add_traces_from_fig(fig_precision_gold, 2, 1, show_legend_flags, "Precision")
    add_traces_from_fig(fig_precision_silver, 2, 2, show_legend_flags, "Precision")
    add_traces_from_fig(fig_f1_gold, 3, 1, show_legend_flags, "F1")
    add_traces_from_fig(fig_f1_silver, 3, 2, show_legend_flags, "F1")

    fig_combined.update_layout(showlegend=True,
                               legend_title_text='Generation Models')
    fig_combined.update_xaxes(title_text="Evaluation Scores")

    # Adjust the position of the subplot titles
    for annotation in fig_combined['layout']['annotations']:
        annotation['yanchor'] = 'bottom'
        annotation['y'] = annotation['y'] + 0.05  # Adjust this value to increase vertical space
        annotation['font'] = dict(size=14)  # You can adjust the font size here if needed


    st.plotly_chart(fig_combined)
