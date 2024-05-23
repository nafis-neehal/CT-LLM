import streamlit as st 
import pandas as pd 
from google.cloud import firestore
import module, module_light

db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")

all_gold_ids = module.get_golddata_ids(sortit=True)

threeshot_example_ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']
main_list = []
shayom_list = []
bowen_list = []

count = 0
for index, id in enumerate(all_gold_ids):

    if id in threeshot_example_ids:
        st.write(f"Trial ID: {id}")
        st.write(f"🔴 :red[This is one of the three dummy trials used for testing and example purposes. \
             Please use the next or previous trial to view actual trial data.]")
        continue

    else:
        st.write(f"Trial ID: {id}")

        #save in gold_main 
        doc_ref = db.collection("All-IDs").document("Gold-100-ids")
        doc_ref.update({
            "id_list": firestore.ArrayUnion([id])
        })
        main_list.append(id)

        #save in shayom
        if count%2 == 0:
            doc_ref = db.collection("All-IDs").document("Shayom-Gold-100-ids")
            doc_ref.update({
                "id_list": firestore.ArrayUnion([id])
            })
            shayom_list.append(id)
            count += 1

        #save in bowen
        else:
            doc_ref = db.collection("All-IDs").document("Bowen-Gold-100-ids")
            doc_ref.update({
                "id_list": firestore.ArrayUnion([id])
            })
            bowen_list.append(id)
            count += 1

st.write(len(main_list))
st.write(len(shayom_list))
st.write(len(bowen_list))
