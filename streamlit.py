import os
import pandas as pd
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from typing import List
import numpy as np

import boto3
import json

# SageMaker endpoint name
ENDPOINT_NAME_MODEL = "flan-t5-demo"
ENDPOINT_NAME_EMBEDDING = "minilm-demo"

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime")

def query_sagemaker_model(question):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME_MODEL,
        ContentType="application/json",
        Body=json.dumps({"inputs": question})
    )
    result = json.loads(response["Body"].read().decode())
    return result[0].get("generated_text", "Error: No response")

def query_sagemaker_embed(question):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME_EMBEDDING,
        ContentType="application/json",
        Body=json.dumps({"inputs": question})
    )
    result = json.loads(response["Body"].read().decode())
    return result



prompt_template = """Answer the following QUESTION based on the CONTEXT
given. If you do not know the answer and the CONTEXT doesn't
contain the answer truthfully say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""



def embed_docs(docs: List[str]) -> List[List[float]]:
    out = query_sagemaker_embed(docs)
    embeddings = np.mean(np.array(out), axis=1)
    return embeddings.tolist()


df_knowledge = pd.read_csv("Amazon_SageMaker_FAQs.csv", header=None, names=["Question", "Answer"])

df_knowledge.drop(["Question"], axis=1, inplace=True)

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY') or 'pcsk_6AFkBu_4x96YeADqKsmBGHZeu7QVjprLhyk5R8nAGs8RNUug2LBkQXyTwnbvJXB76s4RhU'

# configure client
pc = Pinecone(api_key=api_key)
api_key = os.environ.get('PINECONE_API_KEY') or 'pcsk_6AFkBu_4x96YeADqKsmBGHZeu7QVjprLhyk5R8nAGs8RNUug2LBkQXyTwnbvJXB76s4RhU'

pc = Pinecone(api_key=api_key)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'retrieval-augmentation-aws'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=embeddings.shape[1],
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)

batch_size = 2  # can increase but needs larger instance size otherwise instance runs out of memory
vector_limit = 1000

answers = df_knowledge[:vector_limit]

for i in range(0, len(answers), batch_size):
    # find end of batch
    i_end = min(i+batch_size, len(answers))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in answers["Answer"][i:i_end]]
    # create embeddings
    texts = answers["Answer"][i:i_end].tolist()
    embeddings = embed_docs(texts)
    # create records list for upsert
    records = zip(ids, embeddings, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)
    
question = "Which instances can I use with Managed Spot Training in SageMaker?"

# extract embeddings for the questions
query_vec = embed_docs(question)[0]

# query pinecone
res = index.query(vector=query_vec, top_k=5, include_metadata=True)

max_section_len = 1000
separator = "\n"

def construct_context(contexts: List[str]) -> str:
    chosen_sections = []
    chosen_sections_len = 0

    for text in contexts:
        text = text.strip()
        # Add contexts until we run out of space.
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)
    concatenated_doc = separator.join(chosen_sections)
    return concatenated_doc

#context_str = construct_context(contexts=contexts)

#text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)

def rag_query(question: str) -> str:
    # create query vec
    query_vec = embed_docs(question)[0]
    # query pinecone
    res = index.query(vector=query_vec, top_k=5, include_metadata=True)
    # get contexts
    contexts = [match.metadata['text'] for match in res.matches]
    # build the multiple contexts string
    context_str = construct_context(contexts=contexts)
    # create our retrieval augmented prompt
    text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)
    # make prediction
    return text_input

rag_query("Which instances can I use with Managed Spot Training in SageMaker?")

import streamlit as st


# Streamlit UI
st.title("RAG-Powered QA System")
st.write("Ask any question, and the model will generate an answer based on its knowledge and context.")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        rag_prompt = rag_query(question)
        print("AHHHHHHHH" + rag_prompt)
        answer = query_sagemaker_model(rag_prompt)
        st.write("**Answer:**", answer)
    else:
        st.write("Please enter a question.")

