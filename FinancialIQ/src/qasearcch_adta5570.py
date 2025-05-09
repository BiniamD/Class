# -*- coding: utf-8 -*-
"""QASearcch_adta5570.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r8uYnEnwcakBcfFJoY1PSTFj_Aq2CY5X
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install --user --upgrade google-cloud-aiplatform vertexai

import IPython

app=IPython.Application.instance()
app.kernel.do_shutdown(True)

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade langchain
# %pip install --upgrade langchain-core
# %pip install --upgrade langchain-community langchain-google-vertexai
# %pip install --upgrade --quiet langchain-google-community[gcs]

import IPython

app=IPython.Application.instance()
app.kernel.do_shutdown(True)

# Commented out IPython magic to ensure Python compatibility.
# intall : utilities libraires needed for Q&A
! sudo apt -y -qq install tesseract-ocr libtesseract-dev
! sudo apt-get -y -11 install poppler-utils

# %pip install --user --upgrade unstructured pdf2image pytesseract pdfminer.six
# %pip install --user --upgrade pillow-heif opencv-python unstructured-inference pikepdf pypdf

# %pip install --user --upgrade tensorflow_hub tensorflow_text

# %pip install --user --upgrade pi-heif

# %pip install unstructured[inference]

import IPython

app=IPython.Application.instance()
app.kernel.do_shutdown(True)

!pip uninstall torch torchvision torchaudio

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import IPython

app=IPython.Application.instance()
app.kernel.do_shutdown(True)

from posixpath import splitext

import sys

if 'google.colab' in sys.modules:
  from google.colab import auth

  auth.authenticate_user()

from google.cloud import aiplatform
import vertexai

from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)

import langchain

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_google_community import GCSDirectoryLoader, GCSFileLoader

#import Fomr GCP: Vertex AI and Langchain API

from langchain_google_vertexai import VertexAI , VertexAIEmbeddings
from langchain_google_vertexai import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)

import textwrap

import pi_heif

import os

PROJECT_ID = "adta57770"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]": PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
REGION = "us-central1"  # @param {type: "string", placeholder: "[your-region]", isTemplate: true}
if not REGION or REGION == "[your-region]":  REGION = "us-central1"
BUCKET_NAME = "adta5770docs"  # @param {type: "string", placeholder: "[your-bucket-name]", isTemplate: true}
if not BUCKET_NAME or BUCKET_NAME == "[your-bucket-name]": BUCKET_NAME = "gs://[your-bucket-name]"
BUCKET_URL = f"gs://{BUCKET_NAME}"
PREFIX = "sec_filings_pdf"
if not PREFIX or PREFIX == "[your-PREFIX]": PREFIX = "[your-PREFIX]"
PREFIX = "sec_filings_pdf"

print(f"PROJECT_ID: {PROJECT_ID}")
print(f"REGION: {REGION}")
print(f"BUCKET_NAME: {BUCKET_NAME}")
print(f"BUCKET_URL: {BUCKET_URL}")
print(f"PREFIX: {PREFIX}")

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID
    , location=REGION)
    #, staging_bucket=BUCKET_URL)

"""PHASE 4:"""

#---- ADTA 57770: SEMESTER PROJECT: CODING: PHASE 4: CREATE & DEPLOY EMPTY INDEX WITH PUBLIC END POINT


# IMPORTANT NOTES:
# THE CODE IN THIS FILE IS ONLY USED WITH THE FOLLOWING ASSUMPTIONS
# --) The code of all phases of the semester project is included in ONLY ONE Jupyter Notebook document
# --) In other words, the developer complete all phases (1 - 10) in one notebook.
# --) CODE of PHASE 10 is not yet added!

# IMPORTANT NOTES:
# --) The develop have completed coding PHASE 1, 2, and 3 and run the code correctly before starting the code in this file


#---- PHASE 4: STEP 1: Verify the list of currently existing indexes and endpoints
# NOTES: Either none or some already existing.

# First, get the list of indexes that have been created for the vector search system

list_indexes = aiplatform.MatchingEngineIndex.list()

print(f"List of indexes: {list_indexes}")

# Next, get the list of end-points that have been created

list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()

print(f"List of indexes: {list_end_points}")

index_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
for endpoint in index_endpoints:
  print(f"Endpoint: {endpoint.name}")
for deployed_index in endpoint.deployed_indexes:
  print(f"  Deployed Index: {deployed_index.id}")

from google.cloud import aiplatform

# Initialize Vertex AI
#aiplatform.init(project="your-project-id", location="your-region")
aiplatform.init( project=PROJECT_ID  , location=REGION)
# Delete index endpoints
index_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
for endpoint in index_endpoints:
    print(f"Undeploying all indexes for endpoint: {endpoint.name}")
    endpoint.undeploy_all()
    print(f"Deleting endpoint: {endpoint.name}")
    endpoint.delete()

# Delete indexes
indexes = aiplatform.MatchingEngineIndex.list()
for index in indexes:
    print(f"Deleting index: {index.name}")
    index.delete()

print("Cleanup complete.")

"""---- PHASE 4: STEP 2: DELETE ALL currently existing indexes and endpoints

"""

list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()

print(f"List of indexes: {list_end_points}")


#---- PHASE 10: STEP 2: Look for and take notes of numeric end point IDs
# Look for the string of text starting with "resource name"
# Look for a number (long sequence of digits) at the end of the string of text starting with "resource name"
# Copy or take note that number: It is the numeric ID of an end point.
# Copy or take note that number: It is the numeric ID of all end points displayed in the results of listing

# ...

#---- PHASE 10: STEP 3: UNDEPLOYED indexes and DELETE ALL end points

# GET end points
#del_index_endpoint_1 = aiplatform.MatchingEngineIndexEndpoint("3167714989855211520")
#del_index_endpoint_2 = aiplatform.MatchingEngineIndexEndpoint("a numeric ID of some end point")
#del_index_endpoint_3 = aiplatform.MatchingEngineIndexEndpoint("a numeric ID of some end point")
# ... Continue until getting all end points listed above

# Un-deploy indexes and delete end points
#del_index_endpoint_1.undeploy_all()
#del_index_endpoint_1.delete()

#del_index_endpoint_2.undeploy_all()
#del_index_endpoint_2.delete()

#del_index_endpoint_3.undeploy_all()
#del_index_endpoint_3.delete()

# ... Continue until un-deploying all indexes and delete all end points listed above


# Print out the list again to verify --> Should be empty: Nothing is printed out
list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()

print(list_end_points)



#---- PHASE 10: STEP 4: Verify the list of currently existing indexes

# Get the list of indexes that have been created for the vector search system

list_indexes = aiplatform.MatchingEngineIndex.list()

print(f"List of indexes: {list_indexes}")


#---- PHASE 10: STEP 5: Look for and take notes of numeric index IDs
# Look for the string of text starting with "resource name"
# Look for a number (long sequence of digits) at the end of the string of text starting with "resource name"
# Copy or take note that number: It is the numeric ID of an index.
# Copy or take note that number: It is the numeric ID of all indexes displayed in the results of listing

# ...

#---- PHASE 10: STEP 6: DELETE ALL indexes

#del_index_1 = aiplatform.MatchingEngineIndex("6219332546933555200")
#del_index_2 = aiplatform.MatchingEngineIndex("a numeric ID of some index")
#del_index_3 = aiplatform.MatchingEngineIndex("a numeric ID of some index")
# ... Continue to get all indexes

##del_index_1.delete()
#del_index_2.delete()
#del_index_3.delete()
# ... Continue until deleting all indexes

# Verify that all indexes have been deleted --> The list should be empty: Nothing is printed out
list_indexes = aiplatform.MatchingEngineIndex.list()

print(list_indexes)


###========================= AT THIS POINT:
# All indexes have been un-deployed and deleted.
# All end points have been deleted

# COMPLETE SEMESTER PROJECT CODE: PHASE 10

#---- PHASE 4: STEP 4: Create an empty index

# Define constant identifiers: Text names of index and endpoint

# The number of dimensions for the textembedding-005 is 768
DIMENSIONS = 768

A_XYZ_DISPLAY_INDEX_NAME = "FinancialIQ"
A_XYZ_DISPLAY_END_POINT_NAME = "FinancialIQ_END_POINT"
A_XYZ_DEPLOYED_INDEX_ID = "FinancialIQ_INDEX_ID"

#---- PHASE 4: STEP 5: Define text embedding model
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

#---- PHASE 4: STEP 6: Create a new empty vector search index and verify it has been created successfully

a_new_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=A_XYZ_DISPLAY_INDEX_NAME,
    dimensions=DIMENSIONS,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE"
)

if a_new_index:
    print(a_new_index.name)

list_indexes = aiplatform.MatchingEngineIndex.list()

print(list_indexes)

#---- PHASE 4: STEP 7: Create a new index ednpoint and verify it has been created successfully
a_new_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=A_XYZ_DISPLAY_END_POINT_NAME,
    public_endpoint_enabled=True
)

if a_new_endpoint:
    print(a_new_endpoint.name)

list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()

print(list_end_points)

#---- PHASE 4: STEP 8: Deploy the newly created index with a public endpoint

a_new_endpoint = a_new_endpoint.deploy_index(
    index=a_new_index,
    deployed_index_id=A_XYZ_DEPLOYED_INDEX_ID
)

a_new_endpoint.deployed_indexes



###========================= AT THIS POINT:
# A new empty vector search index has been successfully created.
# A new index end point has been successfully created.
# The newly created index has been deployed with a public end point successfully

# COMPLETE SEMESTER PROJECT CODE: PHASE 4 --> READY FOR PHASE 5: EMDBEDDING PDF FILES

from google.cloud import storage
from langchain.document_loaders import PyPDFLoader  # If PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blobs = bucket.list_blobs(prefix=PREFIX)

from langchain_google_community import GCSDirectoryLoader
from langchain_google_community import GCSFileLoader

# ----Initialization

print(f"PROJECT_ID: {PROJECT_ID}")
print(f"REGION: {REGION}")
print(f"BUCKET_NAME: {BUCKET_NAME}")
print(f"BUCKET_URL: {BUCKET_URL}")
print(f"PREFIX: {PREFIX}")

all_documents = []

import warnings
warnings.filterwarnings('ignore', message='.*CropBox missing from /Page.*')

for blob in blobs[:100]: #
  print(str(blob))

  if blob.name.endswith("/") or not blob.name.endswith(".pdf"):
    continue
  print(f"Loading document: {blob.name}")

  loader = GCSFileLoader(
        project_name=PROJECT_ID,
        bucket=BUCKET_NAME,
        blob=blob.name
    )

  documents_from_blob = loader.load()

  document_name = blob.name.split("/")[-1]

  print(f"Document name: {document_name}")

  doc_source_prefix = f"gs://{BUCKET_NAME}"
  doc_source_suffix = "/".join(blob.name.split("/")[1:])
  print(f"Document source suffix: {doc_source_suffix}")
  doc_source = f"{doc_source_prefix}/{doc_source_suffix}"
  print(f"Document source: {doc_source}")

  for document in documents_from_blob:
    document.metadata["source"] = doc_source
    document.metadata["document_name"] = document_name

    all_documents.extend(documents_from_blob)


print(f"Number of documents: {len(all_documents)}")

len(all_documents[:50])

# split the documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

doc_splits = text_splitter.split_documents(all_documents[10:])

for idx,split in enumerate(doc_splits):
  split.metadata["chunk"] = idx

print(f"Number of splits: {len(doc_splits)}")

# Define Cred for Vector store
VSVDB_REGION = "us-central1"
VSVDB_INDEX_NAME = "fiq-vsvd-index"
VSVDB_EMBEDDING_DIR = "fiq-vsvd-bbucket"
VSVDB_DIMENSIONS = 768

#create GCS bucket to store the matching Engine/vector serch index
! set -x && gsutil mb -p $PROJECT_ID -l us-central1 gs://$VSVDB_EMBEDDING_DIR

embeddings = VertexAIEmbeddings(model= "text-embedding-005")

index = aiplatform.MatchingEngineIndex(index_name='3143374001439506432')
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='3645378025333194752')
print (index.name)
print (index_endpoint.name)

vsvectordb = VectorSearchVectorStore.from_components(
    project_id = PROJECT_ID,
    region = VSVDB_REGION,
    gcs_bucket_name =f"gs://{VSVDB_EMBEDDING_DIR}".split("/")[2],
    index_id = index.name,
    embedding = embeddings,
    #documents = doc_splits,
    endpoint_id=index_endpoint.name,
    stream_update=True,

)

texts = [doc.page_content for doc in doc_splits]
metadata = [doc.metadata for doc in doc_splits]

print(f"Number of texts: {len(texts)}")
print(f"Number of metadata: {len(metadata)}")

# Add embeddings to the Vector store
for i in range(0, len(texts),1000):
  vsvectordb.add_texts(texts[i:i+1000], metadata[i:i+1000])

vsvectordb.similarity_search("What are the income of 10x Genomics, Inc.",k=2)

vsvectordb.similarity_search("What are the Risk Factors of DH Enchantment Inc.",k=2)

from langchain_google_vertexai import VertexAI

llm = VertexAI (
    model_name = "gemini-2.5-pro-exp-03-25",
    max_output_tokens = 8192,
    temperature = 0.2,
    top_p = 0.8,
    top_k = 40,
    verbose = True
)

NUMBER_OF_RESULTS = 10

SEARCH_DISTANCE_THRESHOLD = 0.6

retriever = vsvectordb.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k": NUMBER_OF_RESULTS,
        "search_distance": SEARCH_DISTANCE_THRESHOLD
    },
    filters= None,
)

template = """
          **Context:** You are an expert financial analyst assistant. Your task is to analyze and synthesize a specific excerpt from an SEC filing document to directly address a user's query.

          **User Query:**
          "{question}"

          strictly use only the follwing pieces of context to answer the question at the end.Think step by step.
          Do not try to make up an answer:
                          if the context does not contain information about the user's query, just say that you don't know, don't try to make up an answer.
                          if the context is empty, just say that you don't know, don't try to make up an answer.
          **Context:**
          ---
          {context}
          ---

          **Analysis Instructions:**
          1.  **Understand the Query:** What specific information is the user seeking? (e.g., risk factors, financial performance, specific events, definitions).
          2.  **Scan the Excerpt:** Read the provided excerpt carefully.
          3.  **Identify Relevance:** Pinpoint sentences, data points, or statements within the excerpt that *directly* answer or relate to the user's query. Ignore irrelevant information.
          4.  **Synthesize Findings:** Based *only* on the relevant information identified in the excerpt, construct a comprehensive yet concise summary.
              *   Aim for 3-5 clear sentences.
              *   If the query asks for specific details (like risks or numbers), try to include them.
              *   Ensure the summary directly addresses the user's query.
          5.  **Handle Irrelevance:** If the excerpt contains *no information* relevant to the query, explicitly state that "The provided excerpt does not contain information relevant to the query." Do not invent information or summarize unrelated content.

          **Synthesized Summary:**
                            """

#from re import VERBOSE
reteriveal_qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True,
    verbose = True,
    chain_type_kwargs = {
        "prompt": PromptTemplate
        (
                        template = template,
            input_variables = ["context","question"],
        ),
    },
)

# enable for troublshooting
reteriveal_qa.combine_documents_chain.verbose = True
reteriveal_qa.combine_documents_chain.llm_chain.verbose = True
reteriveal_qa.combine_documents_chain.llm_chain.llm.verbose = True

def formatter(result):
  print(f"questions: {result['query']}")
  print("."*80)
  print(f"Answer: {result['result']}")
  print(f"Sources: {result['source_documents']}")
  if "source_documents" in result.keys():
    for idx,ref in enumerate(result["source_documents"]):
      print("."*80)
      print(f"Reference #: {idx}")
      print("."*80)
      if "score" in ref.metadata:
        print(f"Matching Score: {ref.metadata['score']}")
      if "source" in ref.metadata:
        print(f"Source: {ref.metadata['source']}")
      if "document_name" in ref.metadata:
        print(f"Document Name: {ref.metadata['document_name']}")
      print("."*80)
      print(f"Content: \n{ref.page_content}")
  print("."*80)
  print(f"Response:{wrap(result['result'])}")
  print("."*80)

def wrap(s):
  return "\n".join(textwrap.wrap(s, width=120, break_long_words=False))


def ask(
    query,
    qa=reteriveal_qa,
    k=NUMBER_OF_RESULTS,
    search_distance_threshold=SEARCH_DISTANCE_THRESHOLD,
    filters={}
):
  reteriveal_qa.retriever.search_kwargs['search_distance'] = search_distance_threshold
  reteriveal_qa.retriever.search_kwargs['k'] = k
  reteriveal_qa.retriever.search_kwargs['filters'] = filters
  result = reteriveal_qa({"query":query})
  formatter(result)

ask("what is Net income for  ELITE PHARMACEUTICALS INC ?")

filters = {
    "namespace" : "document_name",
    "allwo_list" : ["DH ENCHANTMENT INC_NT 10-Q_2025-02-13.pdf"]
}

ask("What are the Risk Factors of DH Enchantment Inc.?", filters=filters)