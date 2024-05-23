default_msg = {
    "default_system" : "You are a helpful assistant with 10 years of experience in clinical domain and clinical trial design. Answer the question based on the context below and your knowledge on clinical trial design, and clinical domain. You will be given zero or few examples with corresponding answers. Follow a similar pattern to answer the final query to the best of your ability.",
    "default_human" : "Return a list of probable baseline features (seperated by comma, without itemizing or bullet points) that needs to be measured before the trial starts and in each follow up visits. These baseline features are usually found in Table 1 of a trial related publications. Do not give any additional explanations."
}

def get_default_msg():
    return default_msg["default_system"] + "\n\n" + default_msg["default_human"]