import streamlit as st
from gpt4all import GPT4All
import time

# Specify the path to your pre-trained model
model_dir = "path_to_the_model"

# Instantiate GPT4All with your pre-trained model and disable downloads
model = GPT4All(model_dir, allow_download=True)

'Loading...'
iteration=st.empty()
bar=st.progress(0)
for i in range(100):
    iteration.text(f'Loading {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
'Done'

# Generate a response to the user's message
def generate_response(user_message):
    with model.chat_session():
        response = model.generate(prompt=user_message, temp=0)
    # Print the assistant's response
    st.write(f"Assistant: {response}")

# Get a message from the user
user_message = st.text_input("User:")
if user_message:
    generate_response(user_message)

