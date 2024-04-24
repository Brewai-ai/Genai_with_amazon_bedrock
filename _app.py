import streamlit as st
import rockset
from rockset import RocksetClient, Regions
import os
import json
import boto3
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

embeddings = HuggingFaceEmbeddings()

# Initialize Rockset and AWS settings
rockset_key = os.environ.get('ROCKSET_API_KEY')
region = Regions.use1a1
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('REGION_NAME')

def retrieve_information(region, rockset_key, search_query_embedding):
    print("\nRunning Rockset Queries...")
    rs = RocksetClient(api_key=rockset_key, host=region.value)
    api_response = rs.QueryLambdas.execute_query_lambda_by_tag(
        workspace="commons",
        query_lambda="query",
        tag="latest",
        parameters=[
            {"name": "embedding", "type": "array", "value": str(search_query_embedding)}
        ]
    )
    records_list = []
    for record in api_response["results"]:
        records_list.append({"text": record['text']})
    return records_list

def invoke_model(query, retrieved_documents):
    boto3.setup_default_session(aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                region_name=REGION_NAME)
    bedrock = boto3.client(service_name='bedrock-runtime')
    information = ' '.join([doc['text'] for doc in retrieved_documents])
    body = json.dumps({
        "prompt": f"\n\nHuman:You will provide 5 moives names with genres you have to explain them. Question: {query}. \nInformation: {information}\n\nAssistant:",
        "max_tokens_to_sample": 512,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    modelId = 'anthropic.claude-instant-v1'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response['body'].read())
    return response_body.get('completion')

# Streamlit user interface
st.title('Movie Genre Explanation')
genre = st.text_input('Enter a movie genre:', 'Comedy')

if st.button('Generate'):
    search_query_embedding = embeddings.embed_documents(genre)
    records_list = retrieve_information(region, rockset_key, search_query_embedding[0])
    if records_list:
        output = invoke_model(genre, records_list)
        st.text_area("Generated Explanation:", value=output, height=500)
    else:

        st.error("No records found for the provided genre.")

