import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import json
import seaborn as sns
import plotly.express as px
from google.cloud import firestore

def build_zeroshot_prompt(row):

    #prompt structure
    system_message = "You are a helpful assistant with experience in the clinical domain and clinical trial design. \
    You'll be asked queries related to clinical trials. These inquiries will be delineated by a '##Question' heading. \
    Inside these queries, expect to find comprehensive details about the clinical trial structured within specific subsections, \
    indicated by '<>' tags. These subsections include essential information such as the trial's title, brief summary, \
    condition under study, inclusion and exclusion criteria, intervention, and outcomes."

    #baseline measure defintion
    system_message += "In answer to this question, return a list of probable baseline features (separated by commas) of the clinical trial. \
    Baseline features are the set of baseline or demographic characteristics that are assessed at baseline and used in the analysis of the \
    primary outcome measure(s) to characterize the study population and assess validity. Clinical trial-related publications typically \
    include a table of baseline features assessed  by arm or comparison group and for the entire population of participants in the clinical trial."

    #additional instructions
    system_message += "Do not give any additional explanations or use any tags or headers, only return the list of baseline features. "

    #divide row information to generatie the query
    title = row['BriefTitle']
    brief_summary = row['BriefSummary']
    condition = row['Conditions']
    eligibility_criteria = row['EligibilityCriteria']
    intervention = row['Interventions']
    outcome = row['PrimaryOutcomes']

    question = "##Question:\n"
    question += f"<Title> \n {title}\n"
    question += f"<Brief Summary> \n {brief_summary}\n"
    question += f"<Condition> \n {condition}\n"
    question += f"<Eligibility Criteria> \n {eligibility_criteria}\n"
    question += f"<Intervention> \n {intervention}\n"
    question += f"<Outcome> \n {outcome}\n"
    question += "##Answer:\n" 

    st.write(system_message)
    st.write("\n")
    st.write(question)

    #return system_message, question

def build_gold_example_questions_from_row(db):

    ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']

    question = ""

    for id in ids:
        doc_ref = db.collection("Gold-100").document(id)
        doc = doc_ref.get()
        data = doc.to_dict()
        row = data
        question += "##Question:\n"
        question += f"<Title> \n {row['BriefTitle']}\n"
        question += f"<Brief Summary> \n {row['BriefSummary']}\n"
        question += f"<Condition> \n {row['Conditions']}\n"
        question += f"<Eligibility Criteria> \n {row['EligibilityCriteria']}\n"
        question += f"<Intervention> \n {row['Interventions']}\n"
        question += f"<Outcome> \n {row['PrimaryOutcomes']}\n"
        question += "##Answer:\n"
        question += f"{row['Paper_BaselineMeasures']}\n\n"

    return question

def build_three_shot_prompt(row, db):
    #prompt structure
    system_message = "You are a helpful assistant with experience in the clinical domain and clinical trial design. \
    You'll be asked queries related to clinical trials. These inquiries will be delineated by a '##Question' heading. \
    Inside these queries, expect to find comprehensive details about the clinical trial structured within specific subsections, \
    indicated by '<>' tags. These subsections include essential information such as the trial's title, brief summary, \
    condition under study, inclusion and exclusion criteria, intervention, and outcomes."

    #baseline measure defintion
    system_message += "In answer to this question, return a list of probable baseline features (separated by commas) of the clinical trial. \
    Baseline features are the set of baseline or demographic characteristics that are assessed at baseline and used in the analysis of the \
    primary outcome measure(s) to characterize the study population and assess validity. Clinical trial-related publications typically \
    include a table of baseline features assessed  by arm or comparison group and for the entire population of participants in the clinical trial."

    #additional instructions
    system_message += "You will be given three examples. In each example, the question is delineated by '##Question' heading and the corresponding answer is delineated by '##Answer' heading. \
    Follow a similar pattern when you generate answers. Do not give any additional explanations or use any tags or headings, only return the list of baseline features."

    example = build_gold_example_questions_from_row(db)

    #divide row information to generatie the query
    title = row['BriefTitle']
    brief_summary = row['BriefSummary']
    condition = row['Conditions']
    eligibility_criteria = row['EligibilityCriteria']
    intervention = row['Interventions']
    outcome = row['PrimaryOutcomes']

    question = "##Question:\n"
    question += f"<Title> \n {title}\n"
    question += f"<Brief Summary> \n {brief_summary}\n"
    question += f"<Condition> \n {condition}\n"
    question += f"<Eligibility Criteria> \n {eligibility_criteria}\n"
    question += f"<Intervention> \n {intervention}\n"
    question += f"<Outcome> \n {outcome}\n"
    question += "##Answer:\n"

    st.write(system_message)
    st.write("\n")
    st.write(example + question)

    #return system_message, example + question

def get_gpt4_eval_prompt(reference, candidate):
    system = """
You are an expert assistant in the medical domain and clinical trial design. Your task is to determine if a number of candidate features match any features in a given reference list. You need to consider the context and semantics while matching the features.

For each candidate feature:

    1. Identify a matching reference feature based on similarity in context and semantics.
    2. Remember the matched pair.
    3. A reference feature can only be matched to one candidate feature and cannot be further considered for any consecutive matches.
    4. If there are multiple possible matches (i.e. one reference feature can be matched to multiple candidate features or vice versa), choose the most contextually similar one.
    5. Also keep track of which reference and candidate features remain unmatched.

Once the matching is complete, provide the results in a JSON format as follows:
"""
    json_rep = {
        "matched_features": [
            ["<reference feature 1>", "<candidate feature 1>"],
            ["<reference feature 2>", "<candidate feature 2>"]
        ],
        "remaining_reference_features": [
            "<unmatched reference feature 1>",
            "<unmatched reference feature 2>"
        ],
        "remaining_candidate_features": [
            "<unmatched candidate feature 1>",
            "<unmatched candidate feature 2>"
        ]
    }
    system += json.dumps(json_rep, indent=4)

    question = f"\n\nHere is the list of reference features: \n\n"
    ir = 1
    for ref_item in reference:
        question += (
            f"{ir}. {ref_item}\n"
        )
        ir += 1


    question += f"\nCandidate features: \n\n"
    ic = 1
    for can_item in candidate:
        question += (
            f"{ic}. {can_item}\n"
        )
        ic += 1

    #return system, question
    st.write(system)
    st.write("\n")
    st.write(question)



#### Trial2Vec Decoding from Firebase ####
import re

def match_to_score(matches):
    matched_pairs = matches['matched_features']
    remaining_reference_features = matches['remaining_reference_features']
    remaining_candidate_features = matches['remaining_candidate_features']

    precision = len(matched_pairs) / (len(matched_pairs) + len(remaining_candidate_features)) # TP/(TP+FP)
    recall = len(matched_pairs) /  (len(matched_pairs) + len(remaining_reference_features)) #TP/(TP+FN)
    
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall) # F1

    return {"precision": precision, "recall": recall, "f1": f1}

def extract_elements(text):
    # Regex pattern to correctly handle nested commas within parentheses
    # This pattern matches non-parenthetical text or text within balanced parentheses
    pattern = r'\([^()]*\)|[^,]+'

    # Use findall to get all matches, adjusting for nested structure
    matches = re.findall(pattern, text)

    # Clean and combine the elements
    elements = []
    temp = ''
    for match in matches:
        # Continue appending to temp if it starts with an unmatched '('
        if temp.count('(') != temp.count(')'):
            temp += ',' + match
        else:
            if temp:
                elements.append(temp.strip())
            temp = match
    if temp:  # Append the last collected item
        elements.append(temp.strip())

    return elements

def plot_similarity_matrix(similarity_matrix_np, reference, candidate):
    # Plotting the similarity matrix using seaborn's heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(similarity_matrix_np, annot=True, cmap='coolwarm', fmt=".2f",
                     xticklabels=candidate, yticklabels=reference)
    plt.title("Pairwise Cosine Similarities")
    plt.xlabel("Candidate Components")
    plt.ylabel("Reference Components")
    #plt.show()
    st.pyplot(plt)

def clean_string(string):
  """
  Returns a list of items separated by comma in the string.

  Args:
      string: The string to extract items from.
      abc: A list of items to exclude from the final list.

  Returns:
      A list of items separated by comma in the string.
  """

  banned_list = ['continuous', 'categorical', 'custom', 'customized', 'male', 'female', '']

  items = string.split(',')
  final_items = ''
  item_list = []
  for item in items:
    item = item.strip()
    if item.lower() not in banned_list:
      if 'sex' in item.lower():
        if 'Sex' not in item_list:
            final_items += 'Sex, '
            item_list.append('Sex')
      elif 'gender' in item.lower():
        if 'Gender' not in item_list: 
           final_items += 'Gender, '
           item_list.append('Gender')
      elif 'age' in item.lower():
        if 'Age' not in item_list:
           final_items += 'Age, '
           item_list.append('Age')
      else:
        if item not in item_list:
           final_items += item + ', '
           item_list.append(item)

  return final_items[:-2]

def decode_matrix(similarity_matrix_encoded):
    import pickle, base64
    similarity_matrix_serialized = base64.b64decode(similarity_matrix_encoded)
    similarity_matrix = pickle.loads(similarity_matrix_serialized)
    return similarity_matrix


# Function to plot bar charts for a given metric
# def plot_metrics(df, metric):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     sns.barplot(data=df, x='Evaluation_Model', y=metric, hue='Generation_Model', ax=ax)
#     ax.set_xlabel('Evaluation Model')
#     ax.set_ylabel(metric)
#     ax.set_title(f'{metric} by Generation and Evaluation Model')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# Function to plot bar charts using Plotly for a given metric
def plot_metrics(df, metric, dataset_name):
    fig = px.bar(df, x='Generation_Model', y=f'{metric}_mean', color='Evaluation_Model', barmode='group',
                 #error_y=df[f'{metric}_std'],
                 labels={'Evaluation_Model': 'Evaluation Model', f'{metric}_mean': metric},
                 title=f'Average {metric.capitalize()} by Generation and Evaluation Model on {dataset_name} Dataset')
    return fig


def load_firebase(firebase_creds):
    db = firestore.Client.from_service_account_info({
        "type": firebase_creds["type"],
        "project_id": firebase_creds["project_id"],
        "private_key_id": firebase_creds["private_key_id"],
        "private_key": firebase_creds["private_key"].replace('\\n', '\n'),
        "client_email": firebase_creds["client_email"],
        "client_id": firebase_creds["client_id"],
        "auth_uri": firebase_creds["auth_uri"],
        "token_uri": firebase_creds["token_uri"],
        "auth_provider_x509_cert_url": firebase_creds["auth_provider_x509_cert_url"],
        "client_x509_cert_url": firebase_creds["client_x509_cert_url"]
    })

    return db 








