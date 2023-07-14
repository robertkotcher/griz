import os

from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)


#--------------------------------------------------------------------------------------------------------



from langchain.agents import initialize_agent, create_pandas_dataframe_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from langchain.evaluation.loading import load_dataset
from langchain import LLMMathChain
import numpy as np

from sklearn.linear_model import LinearRegression

import openai
import pinecone
import requests
import pandas as pd
import os
import re
import json


OPENAI_API_KEY = "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"

PINECONE_API_KEY = "d0575ab8-506b-4dc6-a65a-395124abb9e5"
PINECONE_ENV = "us-west1-gcp-free"
PINECONE_IDX = "griz-index"

GRIZ_API_URL = "http://api.griz.tech/"
QUERY_URL = GRIZ_API_URL + "bigquery/bigquery-public-data/"

embed_model = "text-embedding-ada-002"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_API_KEY

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

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
                #modified_table_text = f'`bigquery-public-data.{table.replace("`", "")}`'
                modified_table_text = '`bigquery-public-data.{}{}`'.format(table.replace("`", ""), '`')
                splitText[tableIdx] = modified_table_text

    return ''.join(splitText)


def queryResponseToDataFrame(sqlQuery):
    reqBody = {
        "queryText": applyBigQueryProjectIdHack(sqlQuery)
    }
    resp = requests.post(QUERY_URL + "query", json=reqBody).json()
    cols = [col["name"] for col in resp["columns"]]
    rows = resp["rows"]
    return pd.DataFrame(rows, columns=cols)

def generateQueryWithEmbedding(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    pinecone_query = res['data'][0]['embedding']
    pinecone_index = pinecone.GRPCIndex(PINECONE_IDX)
    res = pinecone_index.query(pinecone_query, top_k=2, include_metadata=True)

    record_txt = res["matches"][0]["metadata"]["text"]

    # # generate a query to send to openai
    # query += "---\n\nHere are some relevant tables. Please choose the most relevant one and query it:\n\n"
    # query += record_txt
    # query += "\n\n"
    # query += "An important rule is that if output is SQL, table must be qualified with dataset_id. so <dataset_id>.<table>\n"
    # query += "\nThought: only run a sql query if user asks you to execute the query.\n"
    return record_txt



def askQuestionAboutData(params):
    print("asking further questions about the data!")
    print(params)
    return "none"

generateQueryWithEmbeddingTool = Tool(
    name="generateQueryWithEmbedding",
    func=generateQueryWithEmbedding,
    description="useful when to answer questions about datasets or tables."
)
sqlTool = Tool.from_function(
    name="sqlTool",
    func=queryResponseToDataFrame,
    # docs recommend that for functions with multiple parameters to connect with some separating token
    description="useful when user explicitly asks you to execute a sql query. Input consists of the SQL query to run.",
)
askQuestionAboutDataTool = Tool.from_function(
    name="askQuestionAboutDataTool",
    func=askQuestionAboutData,
    # docs recommend that for functions with multiple parameters to connect with some separating token
    description="useful when asking questions about table data returned from executing a SQL query. The dataframe and original prompt should be passed as a string joined by a semi-colon:   <dataframe>:<prompt>",
)

tools = [generateQueryWithEmbeddingTool]


agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)




#--------------------------------------------------------------------------------------------------------

@app.route("/api/query")
def handle_query():
    query = request.args.get('q')
    res = agent.run(query)
    response = {
        'data': res,
        'status': 200,
    }
    return response

@app.route("/")
def hello_world():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
