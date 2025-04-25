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
from functools import lru_cache
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def ensure_content_cache():
    """Ensure content_cache is initialized in session state"""
    if 'content_cache' not in st.session_state:
        st.session_state.content_cache = {}
    return st.session_state.content_cache

@lru_cache(maxsize=100)
def get_cached_summary(query: str, content_hash: str) -> str:
    """Get cached summary for query and content hash"""
    # Ensure content cache exists
    content_cache = ensure_content_cache()
    
    # Get the result from session state using the content hash
    result = content_cache.get(content_hash)
    if not result:
        logger.error(f"Content not found in cache for hash: {content_hash}")
        return "Error: Content not found in cache"
    
    # Get enhanced context
    context = get_enhanced_context(result, st.session_state.metadata_df)
    
    # Generate reasoning chain
    reasoning_chain = generate_reasoning_chain(query, context)
    
    # Create enhanced prompt
    enhanced_prompt = create_enhanced_prompt(query, context)
    
    # Get initial response
    initial_response = st.session_state.llm.invoke(enhanced_prompt)
    
    # Refine response
    refined_response = refine_response(initial_response, reasoning_chain)
    
    return refined_response

def validate_response(response: str) -> bool:
    """Validate if the LLM response is meaningful"""
    if not response:
        return False
    if len(response.strip()) < 10:
        return False
    if "i don't know" in response.lower():
        return False
    return True

def format_response(response: str) -> str:
    """Format the LLM response for better readability"""
    sections = response.split('\n\n')
    formatted = []
    for section in sections:
        if section.startswith('[Answer]'):
            formatted.append(f"**Answer:** {section[8:].strip()}")
        elif section.startswith('[Supporting Evidence]'):
            formatted.append(f"**Evidence:** {section[19:].strip()}")
        elif section.startswith('[Additional Context]'):
            formatted.append(f"**Context:** {section[18:].strip()}")
        else:
            formatted.append(section)
    return '\n\n'.join(formatted)

def track_response_metrics(query: str, response: str):
    """Track metrics about the response"""
    metrics = {
        'query_length': len(query),
        'response_length': len(response),
        'has_numbers': bool(re.search(r'\d', response)),
        'has_risks': bool(re.search(r'risk|uncertainty|potential', response.lower())),
        'is_meaningful': validate_response(response)
    }
    logger.info(f"Response metrics: {metrics}")

# --- Enhanced RAG Functions ---
def get_enhanced_context(result, metadata_df):
    """Get enhanced context including metadata and temporal information"""
    context = {
        'content': result.page_content,
        'metadata': {
            'company_name': result.metadata.get('companyName', 'Unknown'),
            'form_type': result.metadata.get('formType', 'Unknown'),
            'filing_date': result.metadata.get('filedAt', 'Unknown'),
            'source': result.metadata.get('source', 'Unknown')
        },
        'temporal_context': {
            'is_historical': datetime.now() - datetime.strptime(result.metadata.get('filedAt', ''), '%Y-%m-%d') > timedelta(days=365) if result.metadata.get('filedAt') else False,
            'filing_period': result.metadata.get('filingPeriod', 'Unknown')
        }
    }
    return context

def create_enhanced_prompt(query: str, context: dict) -> str:
    """Create an enhanced prompt with better context handling"""
    return f"""
    **Role:** You are an expert financial analyst with deep knowledge of SEC filings and corporate disclosures.

    **Task:** Analyze the provided SEC filing excerpt and provide a comprehensive answer to the user's query.

    **Document Context:**
    Company: {context['metadata']['company_name']}
    Form Type: {context['metadata']['form_type']}
    Filing Date: {context['metadata']['filing_date']}
    Source: {context['metadata']['source']}
    
    **Historical Context:**
    {'This is a historical filing (over 1 year old). Consider this when interpreting the information.' if context['temporal_context']['is_historical'] else 'This is a recent filing.'}
    
    **Content:**
    {context['content']}

    **User Query:**
    {query}

    **Analysis Instructions:**
    1. First, identify the key aspects of the query that need to be addressed
    2. Then, analyze the document content to find relevant information
    3. Consider the temporal context when interpreting the information
    4. If the information is historical, note any potential changes or updates
    5. Extract and interpret any numerical data or financial metrics
    6. Identify any risks, uncertainties, or important disclosures
    7. Consider the form type when interpreting the information (e.g., 10-K vs 10-Q)

    **Response Structure:**
    [Executive Summary]
    - Brief overview of the key findings
    
    [Detailed Analysis]
    - Comprehensive answer to the query
    - Supporting evidence from the text
    - Interpretation of any numerical data
    - Relevant context and implications
    
    [Additional Context]
    - Temporal considerations
    - Form-specific insights
    - Related disclosures or risks
    - Potential limitations or caveats
    """

def generate_reasoning_chain(query: str, context: dict) -> str:
    """Generate a reasoning chain for better analysis"""
    reasoning_prompt = f"""
    Given the following query and context, generate a reasoning chain:
    
    Query: {query}
    Context: {context}
    
    Step 1: Identify the key aspects of the query
    Step 2: Locate relevant information in the context
    Step 3: Analyze the temporal relevance
    Step 4: Extract and interpret key data points
    Step 5: Consider form-specific implications
    Step 6: Identify any limitations or caveats
    """
    return st.session_state.llm.invoke(reasoning_prompt)

def refine_response(initial_response: str, reasoning_chain: str) -> str:
    """Refine the response using the reasoning chain"""
    refinement_prompt = f"""
    Given the initial response and reasoning chain, refine the response:
    
    Initial Response: {initial_response}
    Reasoning Chain: {reasoning_chain}
    
    Refine the response to:
    1. Ensure all key points from the reasoning chain are addressed
    2. Improve clarity and coherence
    3. Add any missing context or implications
    4. Strengthen the connection between evidence and conclusions
    """
    return st.session_state.llm.invoke(refinement_prompt)

def hash_content(content: str) -> str:
    """Create a hash of the content for caching"""
    return str(hash(content))

# --- Load Environment Variables --- #
load_dotenv()

# --- Configuration Variables --- #
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project-id")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
INDEX_ID = os.getenv("VERTEX_AI_INDEX_ID") # Numeric ID from .env
ENDPOINT_ID = os.getenv("VERTEX_AI_ENDPOINT_ID") # Numeric ID from .env
SOURCE_DOCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "default-source-bucket") # Bucket with original PDFs
PDF_FOLDER = os.getenv("PDF_FOLDER", "default-pdf-folder") # Folder within source bucket
INDEX_DATA_BUCKET = os.getenv("VERTEX_AI_EMBEDDINGS_BUCKET", "default-index-data-bucket") # Bucket used during index creation
EMBEDDING_MODEL = os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-005")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-pro-exp-03-25")
METADATA_CSV_PATH_STR = os.getenv("METADATA_CSV_PATH", "FinancialIQ/documents/metadata.csv")

# Search parameters
NUMBER_OF_RESULTS = 2
SEARCH_DISTANCE_THRESHOLD = 0.7

# --- Log Loaded Configuration --- #
logger.info(f"PROJECT_ID: {PROJECT_ID}")
logger.info(f"LOCATION: {LOCATION}")
logger.info(f"INDEX_ID: {INDEX_ID}")
logger.info(f"ENDPOINT_ID: {ENDPOINT_ID}")
logger.info(f"SOURCE_DOCS_BUCKET: {SOURCE_DOCS_BUCKET}")
logger.info(f"PDF_FOLDER: {PDF_FOLDER}")
logger.info(f"INDEX_DATA_BUCKET: {INDEX_DATA_BUCKET}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"METADATA_CSV_PATH_STR: {METADATA_CSV_PATH_STR}")

# --- Validate Essential Configuration --- #
if not all([PROJECT_ID != "default-project-id", LOCATION, INDEX_ID, ENDPOINT_ID, 
            SOURCE_DOCS_BUCKET != "default-source-bucket", 
            INDEX_DATA_BUCKET != "default-index-data-bucket"]):
    error_message = "One or more critical environment variables are missing or using defaults. Please check your .env file."
    logger.error(error_message)
    st.error(error_message)
    # Optionally stop the app if config is invalid
    # st.stop()

# Helper function to normalize company names for matching (No complex regex)
def normalize_company_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    # Remove common suffixes (with optional period)
    suffixes = [' inc.', ' inc', ' corp.', ' corp', ' corporation', ' llc.', ' llc', 
                ' ltd.', ' ltd', ' limited', ' lp.', ' lp']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break # Remove only one suffix type
    
    # Define punctuation to remove
    punctuation_to_remove = '.,!?"\'()' 
    # Remove punctuation iteratively
    for char in punctuation_to_remove:
        name = name.replace(char, '')
        
    # Remove possessive 's (more robustly)
    name = name.replace("'s", "") # Replace 's anywhere

    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()

# Helper function to extract potential company name prefix from filename
def get_name_prefix_from_filename(filename):
    if not filename:
        return None
    # Try to split by common form types preceded by underscore
    # Ensure specific forms like 10-K/A are checked before simpler ones like 10-K
    common_form_delimiters = ['_10-K/A', '_10-Q/A', '_10-K', '_10-Q', '_8-K']
    prefix = None
    # Check delimiters in order of specificity
    for delimiter in common_form_delimiters:
        if delimiter in filename:
            prefix = filename.split(delimiter)[0].strip()
            logger.info(f"Extracted potential name prefix '{prefix}' using delimiter '{delimiter}' from '{filename}'")
            return prefix

    # Fallback: If no common form delimiter found, try splitting by the last underscore
    if prefix is None and '_' in filename:
        prefix = filename.rsplit('_', 1)[0].strip()
        logger.info(f"Extracted potential name prefix '{prefix}' using last underscore from '{filename}'")
        return prefix

    # Final fallback or if no underscore: Remove extension
    if prefix is None:
        base_name = os.path.basename(filename)
        prefix, _ = os.path.splitext(base_name)
        prefix = prefix.strip()
        logger.warning(f"Could not split by form or underscore, using base filename '{prefix}' as prefix for '{filename}'")
        return prefix

    return None # Should not be reached, but for safety

def verify_bucket_access():
    """Verify access to the SOURCE GCS bucket and list available documents."""
    try:
        client = storage.Client()
        # Use config variable
        bucket = client.get_bucket(SOURCE_DOCS_BUCKET)
        # Use config variable
        blobs = list(bucket.list_blobs(prefix=PDF_FOLDER))
        logger.info(f"Found {len(blobs)} documents in bucket '{SOURCE_DOCS_BUCKET}/{PDF_FOLDER}'")
        logger.info(f"First few documents: {[blob.name for blob in blobs[:3]]}")
        return True
    except Exception as e:
        logger.error(f"Failed to access source bucket '{SOURCE_DOCS_BUCKET}': {str(e)}")
        return False

def verify_index_status():
    """Verify the status of the vector search index."""
    try:
        # Use config variable
        index = aiplatform.MatchingEngineIndex(index_name=INDEX_ID)
        index_resource = index._gca_resource
        logger.info(f"Index name (Display): {index_resource.display_name}")
        logger.info(f"Index description: {index_resource.description}")
        
        # Get the index config
        if hasattr(index_resource, 'index_config'):
            config = index_resource.index_config
            logger.info(f"Index dimensions: {config.dimensions}")
            logger.info(f"Approximate neighbors: {config.approximate_neighbors_count}")
            logger.info(f"Distance measure: {config.distance_measure_type}")
        
        # Use config variable
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_ID)
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
        logger.error(f"Failed to verify index/endpoint status: {str(e)}")
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
    st.session_state.content_cache = {} # Initialize content cache

def initialize_vertex_ai():
    """Initialize Vertex AI using Application Default Credentials. Returns LLM, index name string, endpoint name string, and embeddings object."""
    try:
        # Get default credentials
        credentials, project = default()
        logger.info(f"Successfully obtained credentials for project: {project}")

        # Note: PROJECT_ID and LOCATION are now global config vars
        index_id_numeric = INDEX_ID
        endpoint_id_numeric = ENDPOINT_ID

        # Initialize Vertex AI (using global config vars)
        aiplatform.init(
            project=PROJECT_ID,
            location=LOCATION,
            credentials=credentials
        )
        logger.info("Successfully initialized Vertex AI")
        
        # Verify bucket access (uses global config vars)
        if not verify_bucket_access():
            raise Exception("Failed to verify source document bucket access")

        # Instantiate index and endpoint objects to get their .name attributes
        # Uses global config vars
        index = aiplatform.MatchingEngineIndex(index_name=index_id_numeric)
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id_numeric)
        index_name_str = index.name
        endpoint_name_str = endpoint.name
        logger.info(f"Using Index Name from object: {index_name_str}")
        logger.info(f"Using Endpoint Name from object: {endpoint_name_str}")

        # Verify index status (uses global config vars)
        if not verify_index_status():
            logger.warning("Index verification failed, but continuing with initialization")
        
        # Initialize embeddings (using global config var)
        embeddings = VertexAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        logger.info("Successfully initialized embeddings")
        
        # Initialize LLM (using global config var)
        llm = VertexAI(
            model_name=LLM_MODEL,
            max_output_tokens=8192,
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            verbose=True
        )
        logger.info("Successfully initialized LLM")
        
        # Return components needed for search-time vector store init
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

            # Load metadata CSV after successful init (using global config var)
            try:
                # Use config variable
                st.session_state.metadata_df = pd.read_csv(METADATA_CSV_PATH_STR)
                logger.info(f"Successfully loaded metadata from {METADATA_CSV_PATH_STR}")
            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {METADATA_CSV_PATH_STR}. Cannot display detailed info.")
                st.session_state.metadata_df = None
            except Exception as e:
                 logger.error(f"Error loading metadata from {METADATA_CSV_PATH_STR}: {str(e)}")
                 st.session_state.metadata_df = None

            st.success("Successfully initialized Vertex AI components!")
        else:
             st.error("Failed to initialize one or more Vertex AI components.")

    except Exception as e:
        st.error(f"""
        Authentication/Initialization failed: {str(e)}
        
        To fix this:
        1. Make sure you have run: gcloud auth application-default login
        2. Verify that the project '{PROJECT_ID}' exists and you have access
        3. Check that the index exists in the project
        4. Ensure you have the necessary permissions in the project
        """)

# Search Block
if st.session_state.authenticated:
    # Search interface
    st.subheader("Search Configuration")

    # Search parameters
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of results:", min_value=1, max_value=10, value=NUMBER_OF_RESULTS)
    with col2:
        search_distance = st.slider("Similarity threshold:", min_value=0.0, max_value=1.0, 
                                  value=SEARCH_DISTANCE_THRESHOLD, step=0.1)
     # Filter options
    st.subheader("Filters")
    filter_type = st.selectbox("Filter by:", ["None", "Document Name", "Company", "Form Type"])
    
    filters = None  # Changed from empty dict to None
    if filter_type == "Document Name":
        doc_name = st.text_input("Enter document name (e.g., DH ENCHANTMENT INC_NT 10-Q_2025-02-13.pdf):")
        if doc_name:
            filters = {
                "namespace": "document_name",
                "allow_list": [doc_name]
            }
    elif filter_type == "Company":
        company = st.text_input("Enter company name:")
        if company:
            filters = {
                "namespace": "companyName",
                "allow_list": [company]
            }
    elif filter_type == "Form Type":
        form_type = st.selectbox("Select form type:", ["10-K", "10-Q", "8-K", "NT 10-K", "NT 10-Q"])
        if form_type:
            filters = {
                "namespace": "formType",
                "allow_list": [form_type]
            }
    # Search interface
    query = st.text_input("Enter your search query:",
                        placeholder="e.g., What are the Risk Factors of DH Enchantment Inc.")
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

                # Re-initialize VectorSearchVectorStore for this search (using global config vars, but overriding bucket)
                logger.info("Re-initializing VectorSearchVectorStore for search...")
                vector_store = VectorSearchVectorStore.from_components(
                    project_id=PROJECT_ID,
                    region=LOCATION,
                    gcs_bucket_name='fiq-vsvd-bbucket',
                    index_id=index_name_str,
                    embedding=embeddings,
                    endpoint_id=endpoint_name_str,
                    stream_update=True
                )
                logger.info("VectorSearchVectorStore re-initialized successfully.")

                # Perform the search using the newly initialized vector_store
                results = vector_store.similarity_search(
                    query=query,
                    k=k,
                    score_threshold=search_distance,
                    filters=filters
                )
                logger.info(f"Search filters: {filters}")
                logger.info(f"Search returned {len(results)} results")
                
                if not results:
                    st.warning("No results found matching your criteria. Try adjusting the similarity threshold or rephrasing your query.")
                    st.stop()

                st.subheader("Search Results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}"):
                        # Ensure content cache exists and cache the content
                        content_cache = ensure_content_cache()
                        content_hash = hash_content(result.page_content)
                        content_cache[content_hash] = result

                        # --- 1. AI Summary (from LLM) ---
                        st.write("**AI Summary:**")
                        with st.spinner("Generating summary..."):
                            try:
                                summary = get_cached_summary(query, content_hash)
                                track_response_metrics(query, summary)
                                
                                if not validate_response(summary):
                                    st.warning("The response was not meaningful. Please try rephrasing your query.")
                                else:
                                    formatted_summary = format_response(summary)
                                    st.markdown(formatted_summary)
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                                logger.error(f"LLM error: {str(e)}")

                        # --- 2. Document Info (Improved Lookup with Normalization) ---
                        st.markdown("---")
                        st.write("**Document Info:**")
                        doc_meta = result.metadata
                        doc_name = doc_meta.get('document_name', 'N/A')
                        metadata_df = st.session_state.metadata_df

                        ticker = doc_meta.get('ticker', None)
                        name_prefix_raw = None # Raw prefix from filename
                        normalized_prefix = None # Normalized prefix
                        found_metadata = False
                        meta_row_data = None

                        if ticker and metadata_df is not None:
                            # Direct ticker lookup
                            logger.info(f"Attempting direct lookup for ticker: {ticker}")
                            match = metadata_df[metadata_df['ticker'] == ticker]
                            if not match.empty:
                                meta_row_data = match.iloc[0].to_dict()
                                logger.info(f"Found direct metadata for ticker: {ticker}")
                                found_metadata = True
                            else:
                                logger.warning(f"Direct ticker '{ticker}' not found in metadata.csv")
                        
                        # If direct ticker lookup failed or no ticker in metadata, try prefix lookup with normalization
                        if not found_metadata and metadata_df is not None:
                            name_prefix_raw = get_name_prefix_from_filename(doc_name)
                            if name_prefix_raw:
                                normalized_prefix = normalize_company_name(name_prefix_raw)
                                logger.info(f"Attempting lookup using normalized prefix: {normalized_prefix}")
                                # Apply normalization to DataFrame column for comparison
                                normalized_company_names = metadata_df['companyName'].apply(normalize_company_name)
                                # Case-insensitive starts-with search on normalized names
                                matches = metadata_df[normalized_company_names.str.startswith(normalized_prefix, na=False)]
                                
                                if len(matches) == 1:
                                    meta_row_data = matches.iloc[0].to_dict()
                                    ticker = meta_row_data.get('ticker', 'N/A') # Update ticker if found via prefix
                                    logger.info(f"Found unique normalized match via prefix. Ticker: {ticker}, Company: {meta_row_data.get('companyName')}")
                                    found_metadata = True
                                elif len(matches) > 1:
                                    logger.warning(f"Found multiple company matches ({len(matches)}) for normalized prefix '{normalized_prefix}'. Cannot reliably determine metadata.")
                                else:
                                    logger.warning(f"Found no company matches for normalized prefix '{normalized_prefix}'.")
                            else:
                                logger.warning("Could not extract name prefix from filename.")

                        # Display found metadata or file info
                        if found_metadata and meta_row_data:
                            st.write(f"**Company:** {meta_row_data.get('companyName', 'N/A')}")
                            st.write(f"**Ticker:** {meta_row_data.get('ticker', 'N/A')}")
                            st.write(f"**Form:** {meta_row_data.get('formType', 'N/A')}")
                            st.write(f"**Filed Date:** {meta_row_data.get('filedAt', 'N/A')}")
                            # Optional: Display filing URL as a link
                            filing_url = meta_row_data.get('filing_url', '')
                            if filing_url:
                                st.markdown(f"**Link:** [View Filing]({filing_url})", unsafe_allow_html=True)
                        
                        st.write(f"**Source File:** {doc_name}") # Always show source file
                        
                        # Add informative caption if lookup failed
                        if not found_metadata:
                            if ticker:
                                st.caption(f"(Could not find detailed metadata in CSV for ticker: {ticker})")
                            elif name_prefix_raw:
                                st.caption(f"(Could not find unique company match in CSV for name: {name_prefix_raw})") # Show raw prefix
                            else:
                                st.caption("(Could not determine ticker or name prefix to look up metadata)")

                        # --- 3. Original Excerpt ---
                        st.markdown("---") # Separator
                        st.write("**Original Excerpt:**")
                        # Add placeholder label to fix warning
                        st.text_area("Excerpt Content", value=result.page_content, height=200, disabled=True, label_visibility="collapsed", key=f"excerpt_{i}")

            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                st.error(f"An error occurred during search: {str(e)}")

# Add some helpful information
st.sidebar.title("About")
st.sidebar.info("""
This app allows you to search through SEC filings using semantic search powered by Vertex AI.
Enter your query in natural language and get relevant results from the documents.
""") 