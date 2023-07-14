import os
import base64

from flask import Flask, request, render_template
from flask_cors import CORS

from langchain.agents import initialize_agent, create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from langchain.evaluation.loading import load_dataset
from langchain.chains import LLMChain
from langchain import PromptTemplate
import numpy as np
import subprocess

from sklearn.linear_model import LinearRegression

import openai
import pinecone
import requests
import pandas as pd
import os
import re

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)


#--------------------------------------------------------------------------------------------------------



OPENAI_API_KEY = "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"

PINECONE_API_KEY = "d0575ab8-506b-4dc6-a65a-395124abb9e5"
PINECONE_ENV = "us-west1-gcp-free"
PINECONE_IDX = "griz-index"

GRIZ_API_URL = "http://api.griz.tech/"
QUERY_URL = GRIZ_API_URL + "bigquery/bigquery-public-data/"

embed_model = "text-embedding-ada-002"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_API_KEY

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')

# credit: griztech
# BigQuery has atypical syntax which can be kind of annoying. This is a
# convenience tool that allows you to write queries as expected.
def applyBigQueryProjectIdHack(queryText):
    SCHEMA_QUALIFIERS = ['from', 'join', 'union']
    splitText = re.split(r'(\s+)', queryText.strip())

    for i in range(len(splitText) - 1):
        token = splitText[i].lower()

        if token in SCHEMA_QUALIFIERS:
            tableIdx = i + 1
            while re.match(r'\s+', splitText[tableIdx]):
                tableIdx += 1
            # Skip white spaces

            table = splitText[tableIdx]  # index i+1 is whitespace and i+2 is table
            # the or [] is necessary because match returns None instead of
            # an empty list.
            dotCount = len(re.findall(r'\.', table)) or 0
            # We only want to modify the case the table is expressed
            # as {schema}.{table}. If it's expressed already as
            # `bigquery-public-data.{schema}.{table}` there's no need to
            # modify. If it's expressed without a schema, then it's
            # almost certainly from a named subquery in the text.
            if dotCount == 1:
                modified_table_text = f'`bigquery-public-data.{table.replace("`", "")}`'
                splitText[tableIdx] = modified_table_text

    return ''.join(splitText)

# credit: griztech
def queryResponseToDataFrame(sqlQuery):
    reqBody = {
        "queryText": applyBigQueryProjectIdHack(sqlQuery)
    }
    try:
        resp = requests.post(QUERY_URL + "query", json=reqBody).json()
    except Exception as e: 
        print(e)
    cols = [col["name"] for col in resp["columns"]]
    rows = resp["rows"]
    return pd.DataFrame(rows, columns=cols).to_json()


def get_embedding_text_from_query(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    pinecone_query = res['data'][0]['embedding']
    pinecone_index = pinecone.GRPCIndex(PINECONE_IDX)
    res = pinecone_index.query(pinecone_query, top_k=2, include_metadata=True)

    record_txt = res["matches"][0]["metadata"]["text"]
    return record_txt

def get_answer_from_schema(query, schema):
    prompt = PromptTemplate(
        input_variables=["query", "schema"],
        template="""
        Task: Do you know the answer from just the information provided? If so, please answer. Otherwise, say FALSE (all caps)

        Schema information:
        {schema}

        Query: {query}
        Answer:
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({ "query": query, "schema": schema })
    return response

def get_sql_from_schema(query, schema):
    prompt = PromptTemplate(
        input_variables=["query", "schema"],
        template="""
        Objective:
        - If this query requires us to look at a dataset, generate SQL (Google dialect) to fetch the data. Please wrap the SQL query with ```sql, and ```
        - If this query does not require any further information, output FALSE (all caps)

        Detail: You must provide the table name in format <dataset>.<tablename>

        Schema information:
        {schema}

        Query: {query}
        Answer:
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({ "query": query, "schema": schema })
    print(f"""
        Thought: do we need to query the data?
                {response}
    """)
    return response

def get_python_from_schema_and_data(query, schema, table_data):
    prompt = PromptTemplate(
        input_variables=["query", "schema", "table_data"],
        template="""
        Task: We want to generate Python code to answer the following query
        Query: {query}

        Please follow these rules:
        - If the query has asked to generate a plot, the code should output the file to /tmp/out.png
        - Otherwise, print() the answer to STDOUT.
        - Please wrap the code with ```python, and ```

        Information we have:

        Schema:
        {schema}

        Query results as data frame:
        {table_data}

        Python code:
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({ "query": query, "schema": schema, "table_data": table_data })
    return response


def extract_code(input_string, lang):
    pattern = r"```LANG\n([\s\S]+?)\n```"
    pattern = pattern.replace("LANG", lang)
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def execute_code_with_output(code_string):
    # Run the code as a subprocess
    process = subprocess.Popen(['python', '-c', code_string],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    
    # Capture the stdout and stderr
    stdout, stderr = process.communicate()
    
    # Return the stdout
    return stdout


#--------------------------------------------------------------------------------------------------------

def construct_response(data, status):
    response = {
        'data': data,
        'status': status,
        "is_image": False,
    }
    
    file_path = "/tmp/out.png"    
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            base64_data = base64.b64encode(file.read()).decode("utf-8")
            response["data"] = base64_data
            response["is_image"] = True

    # we'd be doing this in a more sandboxed envrionemtn in the future
    try:
        os.remove(file_path)
    except:
        print("no file to remove. continuing")         

    return response

@app.route("/api/query")
def handle_query():
    query = request.args.get('q')
    print(f"Observation: Got query: {query}")
    
    schema = get_embedding_text_from_query(query)
    print(f"Observation: Found this schema: {schema}")

    # we can answer without seeing anything besides the schema
    a = get_answer_from_schema(query, schema)
    print(f"Thought: Can we answer the question? {a}")
    if a != "FALSE":
        return construct_response(a, 200)

    # we need data, and maybe more things
    sql = get_sql_from_schema(query, schema)
    if sql != "FALSE":
        # first, fetch the data
        sql = extract_code(sql, "sql")
        print(f"Plan: about to execute:")
        print(sql)
        table_data = queryResponseToDataFrame(sql)
        print(table_data)

        # then we can ask a question about it. Maybe the query itself was enough, in
        # which case the python program will be trivial
        script = get_python_from_schema_and_data(query, schema, table_data)
        script = extract_code(script, "python")
        result = execute_code_with_output(script)

        return construct_response(result, 200)
    
    return construct_response("I could not determine the answer to your question!", 200)

@app.route("/")
def hello_world():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
