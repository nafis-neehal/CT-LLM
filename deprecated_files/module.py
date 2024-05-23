import pandas as pd
import random
import requests
import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

import re

# from openai import OpenAI
#from transformers import AutoTokenizer, AutoModel
# from langchain_openai import ChatOpenAI
# from langchain_community.llms import HuggingFaceEndpoint

# # from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain.prompts.prompt import PromptTemplate
# from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate\

# from evaluate import load

from google.cloud import firestore

data = pd.read_csv('data/deprecated/API_1791_CKD_Diab_Obese_Cancer_Hyp.csv')
data_100 = pd.read_csv('data/deprecated/final_K100_shot_response_df.csv')

def get_silverdata_full():
    return data

def get_silverdata_100():
    return data_100

def get_silverdata_ids():
    return data['NCTId'].sort_values().reset_index(drop=True)

def get_golddata_ids(sortit=False):
    dat = pd.read_csv('data/Gold_100_with_llama3_70b.csv')
    if sortit:
        return dat['NCTId'].sort_values().reset_index(drop=True)
    else:
        return dat['NCTId']


def generate_K_shot_examples(df, NCTId, K, seed_value):
    """
    Returns a K number of random rows from main data that are not present in test data.
    These rows are going to be used as K-shot examples for each items in test data.

    Args:
        df: The dataframe to select rows from.
        test_df: The dataframe to check for existing rows.
        k: The number of rows to return.
        seed_value: The seed value for random number generation.

    Returns:
        A dataframe containing the K number of examples for K-shot learning.
    """

    examples = pd.DataFrame()

    random.seed(seed_value)

    #write a function that returns K number of random rows from df that doesn't have NCTId as the given NCTId
    for i in range(K):
        
        random_index = random.randint(0, len(df) - 1)

        while df.iloc[random_index]['NCTId'] == NCTId:
            random_index = random.randint(0, len(df) - 1)

        examples = pd.concat([examples, df.iloc[random_index]], axis=1)

    #st.write(examples.T.reset_index(drop=True))

    return examples.T.reset_index(drop=True)

def few_shot_examples(K, seed, NCTId):

    k_shots = generate_K_shot_examples(data, NCTId, K, seed_value = seed)

    examples = []

    for index, row in k_shots.iterrows():

        trial_info, baseline_features = row_to_info_converter(row)

        answer = f"{baseline_features}"

        examples.append({"trial_info": trial_info, "answer": clean_string(answer)})

    return examples

# def few_shot_examples(K, seed, NCTId, query):

#     k_shots = generate_K_shot_examples(data, NCTId, K, seed_value = seed)

#     examples = []

#     for index, row in k_shots.iterrows():

#         trial_info, baseline_features = row_to_info_converter(row)

#         answer = f"{baseline_features}"

#         examples.append({"trial_info": trial_info, "query": query, "answer": clean_string(answer)})

#     return examples

def get_example_prompt_template():
    example_prompt = PromptTemplate(
        input_variables=["trial_info", "answer"], template="**##Question:** {trial_info} \n\n **##Answer:** {answer}"
    )
    return example_prompt

# def get_example_prompt_template():
#     example_prompt = PromptTemplate(
#         input_variables=["trial_info", "query", "answer"], template="**##Trial Info:** {trial_info} \n\n**##Question:** {query} \n\n**##Answer:** {answer}"
#     )
#     return example_prompt

def get_final_prompt(K, seed, system_message, id):

    examples = few_shot_examples(K=K, seed=seed, NCTId=id)

    final_prompt = FewShotPromptTemplate(
        examples = examples,
        example_prompt= get_example_prompt_template(),
        prefix = system_message,
        suffix = "**##Question:** {trial_info} \n\n **#Answer:** ",
        input_variables = ["trial_info"],
    )

    return final_prompt 

# def get_final_prompt(K, seed, system_message, id, query):

#     examples = few_shot_examples(K=K, seed=seed, NCTId=id, query=query)

#     final_prompt = FewShotPromptTemplate(
#         examples = examples,
#         example_prompt= get_example_prompt_template(),
#         prefix = system_message,
#         suffix = "**##Trial Info:** {trial_info} \n\n **##Question:** {query} \n\n **#Answer:** ",
#         input_variables = ["trial_info", "query"],
#     )

#     return final_prompt


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

  return final_items

def row_to_info_converter(row):
    '''
    Returns a <trial_info> string and a <query> string from the row information.
    Args:
        row: The row information to convert.

    Returns:
        A tuple containing the <trial_info> string and the <query> string.
    '''


    title = row['BriefTitle']
    brief_summary = row['BriefSummary']
    condition = row['Conditions']
    inclusion = row['InclusionCriteria']
    exclusion = row['ExclusionCriteria']
    intervention = row['Interventions']
    outcome = row['PrimaryOutcomes']

    trial_info = (
       f"\n<Title> - {title} \n"
       f"<Brief Summary> - {brief_summary} \n"
       f"<Condition> - {condition}\n"
       f"<Inclusion Criteria> - {inclusion}\n"
       f"<Exclusion Criteria> - {exclusion}\n"
       f"<Intervention> - {intervention}\n"
       f"<Outcome> - {outcome}"
    )

    baseline = row['BaselineMeasures']

    # query = "Return a list of probable baseline features (seperated by comma, without itemizing or bullet points) that needs to\
    # be measured before the trial starts and in each follow up visits. These baseline\
    # features are usually found in Table 1 of a trial related publications. Don't give any additional explanations."

    return (trial_info, baseline)

def print_trial(df, print_responses=False, show_id=False):
    if show_id:
       st.write("**Trial ID:**", df['NCTId'])
    st.write("**Brief Title:**", df['BriefTitle'])
    st.write("**Condition:**", df['Conditions'])
    st.write("**Brief Summary:**", df['BriefSummary'])
    st.write("**Eligibility Criteria**")
    st.write("**Inclusion Criteria:**\n", df['InclusionCriteria'])
    st.write("**Exclusion Criteria:**\n", df['ExclusionCriteria'])
    st.write("**Primary Outcome Measure:**", df['PrimaryOutcomes'])
    st.write("**Intervention Name:**", df['Interventions'])
    st.write("**Study Type:**", df['StudyType'])
    st.write("**Baseline Measures**", clean_string(df['BaselineMeasures']))

    if print_responses:
       st.write("**Zero-Shot:**", clean_string(df['ZeroShot']))
       st.write("**One-Shot:**", clean_string(df['OneShot']))
       st.write("**Two-Shot:**", clean_string(df['TwoShot']))
       st.write("**Three-Shot:**", clean_string(df['ThreeShot']))

#---------------- GPT-4 Eval ----------------#

def get_gpt4_eval_prompt(base, response):

    system = "You are a helpful assistant with 10 years of experience in clinical domain and clinical trial design.\
        Answer the question based on your knowledge on clinical trial design, and clinical domain."
    
    system += f" You will be asked about a number of features and for each of them you have to determine if they match any of the features in the base list."
    system += f" Remember to consider the context and semantics of the features while considering a feature."
    system += f" Remember your choice for every feature and the feature it got matched to in the base list if there is a match. \n\n"
    system += f"Here is the base list - {base}\n\n"

    prompt = ""

    for item in response:
        prompt += (
            f"Does this feature '{item}' match?\n\n"   
        )
    
    prompt += "Reply a JSON object in this format: \n\n {matched: <this is a list of tuples> [(item 1 that matched in response, which feature it matched to in base), \
                                                              (item 2 that matched in response, which feature it matched to in base)],\
                                                    unmatched_response: <this is a list of items> [item 1 that did not match to base, item 2 that did not match to base],\
                                                    unmatched_base: <this is a list of items> [item 1 that did not match to response, item 2 that did not match to response]"


    return system, prompt
   

def get_gpt4_eval_score(system, prompt):

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        seed = 1111,
        response_format={"type": "json_object"},
        messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
        ]
    )

    #st.write(completion.choices[0].message.content)
    return response.choices[0].message.content

def get_gpt35_eval_score(system, prompt):

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        seed = 1111,
        response_format={"type": "json_object"},
        messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
        ]
    )

    #st.write(completion.choices[0].message.content)
    return response.choices[0].message.content

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

# Function to get embeddings for a list of components
def get_embeddings(text_list, tokenizer, model):
    # Tokenize the text and convert to IDs
    inputs = tokenizer(text_list, padding=True, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    # Get the mean of the last hidden state to represent each token
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to plot the similarity matrix
def plot_similarity_matrix(similarities, reference_tokens, candidate_tokens):
    similarities = similarities.detach().numpy()
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a heatmap
    sns.heatmap(similarities, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=candidate_tokens, yticklabels=reference_tokens, ax=ax)
    
    # Set the title and labels
    ax.set_title("Pairwise Cosine Similarities")
    ax.set_ylabel("Reference Components")
    ax.set_xlabel("Candidate Components")
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def get_match_json(similarity_matrix, reference, candidate, threshold):
    # Finding matches and managing unmatched elements
    matched = []
    unmatched_reference = set(reference)
    unmatched_candidate = set(candidate)

    # Iterate over each candidate and find the best matching reference above the threshold
    for j, can_ele in enumerate(candidate):
        max_sim, max_index = torch.max(similarity_matrix[:, j], 0)
        if max_sim.item() > threshold:
            ref_ele = reference[max_index]
            matched.append((can_ele, ref_ele)) #[candidate, reference]
            if ref_ele in unmatched_reference:
                unmatched_reference.remove(ref_ele)
            if can_ele in unmatched_candidate:
                unmatched_candidate.remove(can_ele)

    # Create the result JSON
    result = {
        "matched": matched,
        "unmatched_reference": list(unmatched_reference),
        "unmatched_candidate": list(unmatched_candidate)
    }

    return result

# def get_bert_eval_score(base, response):
#     # ref_str = "Age, Gender, Ethnicity (NIH/OMB), Race/Ethnicity, Region of Enrollment, \
#     # Previous cardiovascular disease (CVD) event, Glycated hemoglobin, Blood pressure, Cholesterol, \
#     # Triglycerides, Diabetes duration,"

#     # candidate_str = "Age, gender, body mass index (BMI), fasting plasma glucose, HbA1c, \
#     # blood pressure, lipid profile (including LDL, HDL, total cholesterol, triglycerides), \
#     # history of cardiovascular disease (CVD), history of diabetes mellitus, current medication use, \
#     # smoking status, renal function tests (e.g., serum creatinine, eGFR)"

#     reference_tokens = extract_elements(base)
#     candidate_tokens = extract_elements(response)

#     print(reference_tokens)
#     print(candidate_tokens)

#     # # Load a pretrained BERT model and tokenizer
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     # model = BertModel.from_pretrained('bert-base-uncased')

#     # tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
#     # model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

#     # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#     # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#     tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
#     model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

#     # Get embeddings for reference and candidate tokens
#     reference_embeddings = get_embeddings(reference_tokens, tokenizer, model)
#     candidate_embeddings = get_embeddings(candidate_tokens, tokenizer, model)

#     # Calculate pairwise cosine similarities
#     similarities = torch.mm(reference_embeddings, candidate_embeddings.T)
#     similarities = F.cosine_similarity(reference_embeddings[:, None], candidate_embeddings[None, :], dim=2)

#     # Call the plotting function
#     plot_similarity_matrix(similarities, reference_tokens, candidate_tokens)


def get_bert_eval_score2(ref, can, threshold):

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    reference_tokens = extract_elements(ref)
    candidate_tokens = extract_elements(can)

    # # Assume model.sentence_vector is a method that returns a tensor of shape [n, 128] for n sentences
    # reference_tensor = model.sentence_vector(reference).cpu()
    # candidate_tensor = model.sentence_vector(candidate).cpu()

    # Get embeddings for reference and candidate tokens
    reference_embeddings = get_embeddings(reference_tokens, tokenizer, model)
    candidate_embeddings = get_embeddings(candidate_tokens, tokenizer, model)

    # Normalize the tensors
    reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)

    # Compute the similarity matrix
    similarity_matrix = torch.mm(reference_embeddings, candidate_embeddings.transpose(0, 1))

    result = get_match_json(similarity_matrix, reference_tokens, candidate_tokens, threshold)
    plot_similarity_matrix(similarity_matrix, reference_tokens, candidate_tokens)

    st.json(result)
    
    # return json.dumps(result, indent=2)


    



def get_list_from_string(string):
    """
    Returns a list of items separated by comma in the string.
    
    Args:
        string: The string to extract items from.
    
    Returns:
        A list of items separated by comma in the string.
    """
    
    items = string.split(',')
    final_items = []
    for item in items:
        if item != " ":
           final_items.append(item.strip())
    
    return final_items


def print_gpt_response(json_obj):

    # Flatten the JSON object
    flattened_data = {
        "matched": [tuple(sublist) for sublist in json_obj["matched"]],
        "unmatched_response": json_obj["unmatched_response"],
        "unmatched_base": json_obj["unmatched_base"]
    }

    # Convert the flattened data to a DataFrame
    df = pd.DataFrame.from_dict(flattened_data, orient='index')

    # Transpose the DataFrame so that the keys of the JSON object become the column headers
    df = df.transpose()

    # Display the DataFrame as a table in Streamlit
    st.table(df)

# ----------------- Database -----------------#
def format_firestore_timestamp(timestamp):
    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_timestamp



