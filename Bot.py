from gpt4all import GPT4All

# Specify the path to your pre-trained model
model_dir = "fmodel.bin"

# Instantiate GPT4All with your pre-trained model and disable downloads
model = GPT4All(model_dir, allow_download=False)

# Print the default system and prompt templates
print("Default system template:", repr(model.config['systemPrompt']))
print("Default prompt template:", repr(model.config['promptTemplate']))
print()

# Start a conversation with the bot
while True:
    # Get a message from the user
    user_message = input("User: ")

    # If the user types 'exit', end the conversation
    if user_message.lower() == "exit":
        break

    # Generate a response to the user's message
    with model.chat_session():
        response = model.generate(prompt=user_message, temp=0)
        print("Session system template:", repr(model.current_chat_session[0]['content']))
        print("Session prompt template:", repr(model._current_prompt_template))
    
    # Print the assistant's response
    print(f"Assistant: {response}")

