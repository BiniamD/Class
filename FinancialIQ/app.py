import os
import streamlit as st
from dotenv import load_dotenv
import google.cloud.aiplatform as aiplatform
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_google_vertexai import VectorSearchVectorStore
from google.auth import default
from google.cloud import storage
import logging
import pandas as pd
import re # For parsing filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to extract ticker (Updated Logic)
def get_ticker_from_filename(filename):
    if not filename:
        return None
    # Get the base name (e.g., "TICKER.pdf")
    base_name = os.path.basename(filename)
    # Get the name without extension (e.g., "TICKER")
    ticker, _ = os.path.splitext(base_name)
    # Return the result (potentially converting to upper if needed, but try as is first)
    logger.info(f"Extracted ticker '{ticker}' from filename '{filename}'") # Add logging
    return ticker

# Load environment variables
load_dotenv()

def verify_bucket_access():
    """Verify access to the GCS bucket and list available documents."""
    try:
        client = storage.Client()
        bucket = client.get_bucket('adta5770docs')
        blobs = list(bucket.list_blobs(prefix='sec_filings_pdf'))
        logger.info(f"Found {len(blobs)} documents in bucket 'adta5770docs'")
        logger.info(f"First few documents: {[blob.name for blob in blobs[:3]]}")
        return True
    except Exception as e:
        logger.error(f"Failed to access bucket: {str(e)}")
        return False

def verify_index_status():
    """Verify the status of the vector search index."""
    try:
        # Get the index by ID
        index = aiplatform.MatchingEngineIndex(index_name='3143374001439506432')
        
        # Get index details using the correct API
        index_resource = index._gca_resource
        logger.info(f"Index name: {index_resource.display_name}")
        logger.info(f"Index description: {index_resource.description}")
        
        # Get the index config
        if hasattr(index_resource, 'index_config'):
            config = index_resource.index_config
            logger.info(f"Index dimensions: {config.dimensions}")
            logger.info(f"Approximate neighbors: {config.approximate_neighbors_count}")
            logger.info(f"Distance measure: {config.distance_measure_type}")
        
        # Get endpoint details
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='3645378025333194752')
        endpoint_resource = endpoint._gca_resource
        logger.info(f"Endpoint name: {endpoint_resource.display_name}")
        logger.info(f"Endpoint description: {endpoint_resource.description}")
        
        # Check deployed indexes
        if endpoint_resource.deployed_indexes:
            logger.info("Deployed indexes:")
            for deployed_index in endpoint_resource.deployed_indexes:
                logger.info(f"  - ID: {deployed_index.id}")
                logger.info(f"    Display name: {deployed_index.display_name}")
        else:
            logger.warning("No indexes deployed to endpoint")
        
        return True
    except Exception as e:
        logger.error(f"Failed to verify index status: {str(e)}")
        return False

# Streamlit app
st.title("FinancialIQ Vector Search")
st.write("Search through SEC filings using semantic search")

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.llm = None
    st.session_state.index_name_str = None
    st.session_state.endpoint_name_str = None
    st.session_state.embeddings = None # Store embeddings object too
    st.session_state.metadata_df = None # Add state for metadata

def initialize_vertex_ai():
    """Initialize Vertex AI using Application Default Credentials. Returns LLM, index name string, endpoint name string, and embeddings object."""
    try:
        # Get default credentials
        credentials, project = default()
        logger.info(f"Successfully obtained credentials for project: {project}")
        
        project_id = 'adta57770'
        location = 'us-central1'
        index_id_numeric = '3143374001439506432'
        endpoint_id_numeric = '3645378025333194752'

        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        logger.info("Successfully initialized Vertex AI")
        
        # Verify bucket access
        if not verify_bucket_access():
            raise Exception("Failed to verify bucket access")

        # Instantiate index and endpoint objects to get their .name attributes
        index = aiplatform.MatchingEngineIndex(index_name=index_id_numeric)
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id_numeric)
        index_name_str = index.name
        endpoint_name_str = endpoint.name
        logger.info(f"Using Index Name from object: {index_name_str}")
        logger.info(f"Using Endpoint Name from object: {endpoint_name_str}")

        # Verify index status (using numeric IDs for verification function)
        if not verify_index_status(): # verify_index_status still uses numeric IDs internally
            logger.warning("Index verification failed, but continuing with initialization")
        
        # Initialize embeddings
        embeddings = VertexAIEmbeddings(
            model="text-embedding-005"
        )
        logger.info("Successfully initialized embeddings")
        
        # Initialize LLM
        llm = VertexAI(
            model_name="gemini-2.5-pro-exp-03-25",
            max_output_tokens=8192,
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            verbose=True
        )
        logger.info("Successfully initialized LLM")
        
        # Don't initialize vector store here, just return components
        logger.info("Initialization complete, returning components.")
        return llm, index_name_str, endpoint_name_str, embeddings
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        st.error(f"Initialization failed: {str(e)}")
        return None, None, None, None

# Authentication Block
if not st.session_state.authenticated:
    try:
        # First, try to authenticate
        credentials, project = default()
        st.info("Successfully obtained credentials")
        
        # Then initialize Vertex AI components
        llm, index_name_str, endpoint_name_str, embeddings = initialize_vertex_ai()
        if llm and index_name_str and endpoint_name_str and embeddings:
            st.session_state.authenticated = True
            st.session_state.llm = llm
            st.session_state.index_name_str = index_name_str
            st.session_state.endpoint_name_str = endpoint_name_str
            st.session_state.embeddings = embeddings

            # Load metadata CSV after successful init
            try:
                metadata_path = "FinancialIQ/documents/metadata.csv"
                st.session_state.metadata_df = pd.read_csv(metadata_path)
                logger.info(f"Successfully loaded metadata from {metadata_path}")
            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {metadata_path}. Cannot display detailed info.")
                st.session_state.metadata_df = None # Ensure it's None if not found
            except Exception as e:
                 logger.error(f"Error loading metadata: {str(e)}")
                 st.session_state.metadata_df = None

            st.success("Successfully initialized Vertex AI components!")
        else:
             st.error("Failed to initialize one or more Vertex AI components.")

    except Exception as e:
        st.error(f"""
        Authentication/Initialization failed: {str(e)}
        
        To fix this:
        1. Make sure you have run: gcloud auth application-default login
        2. Verify that the project 'adta57770' exists and you have access
        3. Check that the index exists in the project
        4. Ensure you have the necessary permissions in the project
        """)

# Search Block
if st.session_state.authenticated:
    # Search interface
    query = st.text_input("Enter your search query:",
                        placeholder="e.g., What are the Risk Factors of DH Enchantment Inc.")

    # REMOVE SLIDER: Force k=1
    # k = st.slider("Number of results to return:", min_value=1, max_value=10, value=2)
    k = 1

    if query:
        with st.spinner("Searching..."):
            try:
                logger.info(f"Performing search for query: {query}")

                # Retrieve components from session state
                llm = st.session_state.llm
                index_name_str = st.session_state.index_name_str
                endpoint_name_str = st.session_state.endpoint_name_str
                embeddings = st.session_state.embeddings
                metadata_df = st.session_state.metadata_df # Retrieve metadata df

                if not (llm and index_name_str and endpoint_name_str and embeddings):
                    st.error("Search error: Components not found in session state.")
                    st.stop()

                # Re-initialize VectorSearchVectorStore for this search
                logger.info("Re-initializing VectorSearchVectorStore for search...")
                vector_store = VectorSearchVectorStore.from_components(
                    project_id='adta57770',
                    region='us-central1',
                    gcs_bucket_name='fiq-vsvd-bbucket', # Align with index creation bucket
                    index_id=index_name_str,
                    embedding=embeddings,
                    endpoint_id=endpoint_name_str,
                    stream_update=True
                )
                logger.info("VectorSearchVectorStore re-initialized successfully.")

                # Perform the search using the newly initialized vector_store
                results = vector_store.similarity_search(
                    query=query,
                    k=k
                )
                logger.info(f"Search returned {len(results)} results")
                
                st.subheader("Search Results")
                for i, result in enumerate(results, 1): # Loop will only run once if k=1
                    with st.expander(f"Result {i}"):

                        # --- 1. AI Summary (from LLM) ---
                        st.write("**AI Summary:**")
                        with st.spinner("Generating summary..."):
                            # Use the more elaborative prompt template
                            summary_prompt = f"""
                            **Context:** You are an expert financial analyst assistant. Your task is to analyze and synthesize a specific excerpt from an SEC filing document to directly address a user's query.

                            **User Query:**
                            "{query}"

                            **Retrieved SEC Filing Excerpt:**
                            ---
                            {result.page_content}
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
                            summary = st.session_state.llm.invoke(summary_prompt)
                            st.write(summary) # Display only the LLM output

                        # --- 2. Document Info (from metadata) ---
                        st.markdown("---") # Separator
                        st.write("**Document Info:**")
                        doc_name = result.metadata.get('document_name', 'N/A')
                        ticker = get_ticker_from_filename(doc_name)
                        found_metadata = False
                        if ticker and metadata_df is not None:
                            meta_row = metadata_df[metadata_df['ticker'] == ticker]
                            if not meta_row.empty:
                                meta_row = meta_row.iloc[0] # Get first match
                                st.write(f"**Company:** {meta_row.get('companyName', 'N/A')}")
                                st.write(f"**Form:** {meta_row.get('formType', 'N/A')}")
                                st.write(f"**Filed Date:** {meta_row.get('filedAt', 'N/A')}")
                                # Could add filing_url link here too
                                # st.write(f"**Link:** [{meta_row.get('filing_url', 'N/A')}]({meta_row.get('filing_url', '')})")
                                found_metadata = True

                        st.write(f"**Source File:** {doc_name}") # Always show source file
                        if not found_metadata and ticker:
                            st.caption(f"(Could not find detailed metadata for ticker: {ticker})")
                        elif not ticker:
                             st.caption(f"(Could not extract ticker from filename)")


                        # --- 3. Original Excerpt ---
                        st.markdown("---") # Separator
                        st.write("**Original Excerpt:**")
                        # Add a unique key based on the loop index 'i'
                        st.text_area("", value=result.page_content, height=200, disabled=True, label_visibility="collapsed", key=f"excerpt_{i}")

            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                st.error(f"An error occurred during search: {str(e)}")

# Add some helpful information
st.sidebar.title("About")
st.sidebar.info("""
This app allows you to search through SEC filings using semantic search powered by Vertex AI.
Enter your query in natural language and get relevant results from the documents.
""") 