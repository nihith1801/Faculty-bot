from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import csv
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow requests from 'http://localhost:3000'
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 0.2 * 1024 * 1024  # 200kb

@app.route('/chat_csv', methods=['POST'])
def chat_csv():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500
    query = request.form.get('query')

    # Specify the path to the CSV file on your local machine
    file_path = 'faculty.csv'

    # Create the CSV agent
    llm = OpenAI()
    csv_agent = create_csv_agent(llm, file_path)

    # Read the CSV file and handle potential issues with the 'Name' key and empty rows
    data = []
    for row in csv_agent:
        if 'Name' in row and row['Name'] and re.search(query.lower(), row['Name'].lower()):
            data.append(row)

    # Format the data to include all the fields
    formatted_data = []
    for row in data:
        formatted_row = {
            'Name': row.get('Name', ''),
            'Image': "<img src={}></>" % row.get('Image', ''),
            'Email': row.get('Email', ''),
            'Social contact': row.get('Social contact', ''),
            'Title': row.get('Title', ''),
            'Research Interest': row.get('Research Interest', ''),
            'Administrative Responsibility': row.get('Administrative Responsibility', ''),
            'Profile': row.get('Profile', '')
        }
        formatted_data.append(formatted_row)

    # Create the OpenAI agent and get the response
    agent = create_csv_agent(
        OpenAI(temperature=0, max_tokens=500), file_path, verbose=True)

    prompt = query #"Which product line had the lowest average price"

    if prompt is None or prompt == "":
        return jsonify({"error": "No user question provided"}), 400
    
    # Add a prefix to the prompt to request a point format answer
    prompt = "Please provide the answer in point format: " + prompt

    response = agent.run(prompt)
    
    # You can format the response as needed, e.g., convert to JSON
    response_json = {"answer": response}
    
    return jsonify(formatted_data, response_json), 200

if __name__ == "__main__":
    app.run(debug=True)
