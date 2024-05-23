import streamlit as st 
import pandas as pd 
from google.cloud import firestore
import module_lite

#db = firestore.Client.from_service_account_json("ct-llm-firebase-key.json")
#all_gold_ids = module.get_golddata_ids(sortit=True)

firebase_creds = st.secrets["firebase"]
db = module_lite.load_firebase(firebase_creds)
id_ref = db.collection("All-IDs").document("Gold-100-ids")
id_dat = id_ref.get().to_dict()
all_gold_ids = id_dat['id_list']

threeshot_example_ids = ['NCT00000620', 'NCT01483560', 'NCT04280783']
main_list = []
shayom_list = []
bowen_list = []
vibha_list = []

count = 0
for index, id in enumerate(all_gold_ids):

    if id in threeshot_example_ids:
        # st.write(f"Trial ID: {id}")
        # st.write(f"🔴 :red[This is one of the three dummy trials used for testing and example purposes. \
        #      Please use the next or previous trial to view actual trial data.]")
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
        if index%2 == 0:
            doc_ref = db.collection("All-IDs").document("Shayom-Gold-100-ids")
            doc_ref.update({
                "id_list": firestore.ArrayUnion([id])
            })
            shayom_list.append(id)
            #count += 1

        #save in bowen
        else:
            doc_ref = db.collection("All-IDs").document("Bowen-Gold-100-ids")
            doc_ref.update({
                "id_list": firestore.ArrayUnion([id])
            })
            bowen_list.append(id)
            #count += 1

        #save in vibha
        if index%3 == 0:
            doc_ref = db.collection("All-IDs").document("Vibha-Gold-100-ids")
            doc_ref.update({
                "id_list": firestore.ArrayUnion([id])
            })
            vibha_list.append(id)
            count += 1
        if count == 20:
            break 

# st.write(len(main_list))
# st.write(len(shayom_list))
# st.write(len(bowen_list))
