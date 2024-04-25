import pandas as pd
import random
import requests
import streamlit as st
import numpy as np

from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

# from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate

data = pd.read_csv('data/API_1791_CKD_Diab_Obese_Cancer_Hyp.csv')
data_100 = pd.read_csv('data/final_K100_shot_response_df.csv')

def get_silverdata_full():
    return data

def get_silverdata_100():
    return data_100

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

def few_shot_examples(K, seed, NCTId, query):

    k_shots = generate_K_shot_examples(data, NCTId, K, seed_value = seed)

    examples = []

    for index, row in k_shots.iterrows():

        trial_info, baseline_features = row_to_info_converter(row)

        answer = f"{baseline_features}"

        examples.append({"trial_info": trial_info, "query": query, "answer": clean_string(answer)})

    return examples


def get_example_prompt_template():
    example_prompt = PromptTemplate(
        input_variables=["trial_info", "query", "answer"], template="**##Trial Info:** {trial_info} \n\n**##Question:** {query} \n\n**##Answer:** {answer}"
    )
    return example_prompt

def get_final_prompt(K, seed, system_message, id, query):

    examples = few_shot_examples(K=K, seed=seed, NCTId=id, query=query)

    final_prompt = FewShotPromptTemplate(
        examples = examples,
        example_prompt= get_example_prompt_template(),
        prefix = system_message,
        suffix = "**##Trial Info:** {trial_info} \n\n **##Question:** {query} \n\n **#Answer:** ",
        input_variables = ["trial_info", "query"],
    )

    return final_prompt


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

def print_trial(df, print_responses=False):
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

    prompt = ""

    for item in response:
        prompt += (
            f"Does this feature '{item}' match (contextually/semantically) any of the features in this list of features - {base}. "
            f"Remember your choice for this feature and the feature it got matched to if there is a match. \n\n"
        )
    
    prompt += "Reply a JSON object in this format: {matched: <this is a list of tuples> [(item 1 that matched, which keyword it matched to in base), \
                                                              (item 2 that matched, which keyword it matched to in base)],\
                                                    unmatched: <this is a list of items> [item 1 that did not match, item 2 that did not match]}"


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



