from google.cloud import aiplatform
from langchain.embeddings import VertexAIEmbeddings

# Initialize Vertex AI with your project details
def initialize_vertex_ai(project_id, location="us-central1"):
    """
    Initialize Vertex AI with project details
    """
    aiplatform.init(project=project_id, location=location)
    print(f"Vertex AI initialized with project: {project_id}")

def phase_4(project_id, location="us-central1"):
    """
    Phase 4: Create and deploy empty index with public endpoint
    """
    print("Starting Phase 4: Create and deploy empty index with public endpoint")
    
    # Initialize Vertex AI
    initialize_vertex_ai(project_id, location)
    
    # Step 1: Verify existing indexes and endpoints
    print("\nStep 1: Verifying existing indexes and endpoints")
    list_indexes = aiplatform.MatchingEngineIndex.list()
    print(f"List of indexes: {list_indexes}")
    
    list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()
    print(f"List of endpoints: {list_end_points}")
    
    # Step 2: Delete existing indexes and endpoints
    print("\nStep 2: Deleting existing indexes and endpoints")
    for endpoint in list_end_points:
        endpoint.undeploy_all()
        endpoint.delete()
    
    for index in list_indexes:
        index.delete()
    
    # Step 3: Verify no indexes or endpoints exist
    print("\nStep 3: Verifying no indexes or endpoints exist")
    list_indexes = aiplatform.MatchingEngineIndex.list()
    list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()
    print(f"List of indexes: {list_indexes}")
    print(f"List of endpoints: {list_end_points}")
    
    # Step 4: Define constants
    DIMENSIONS = 768  # Dimensions for textembedding-005
    A_XYZ_DISPLAY_INDEX_NAME = "semester_project_index"
    A_XYZ_DISPLAY_END_POINT_NAME = "semester_project_endpoint"
    A_XYZ_DEPLOYED_INDEX_ID = "semester_project_deployed_index"
    
    # Step 5: Define text embedding model
    print("\nStep 5: Defining text embedding model")
    embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")
    
    # Step 6: Create new empty vector search index
    print("\nStep 6: Creating new empty vector search index")
    a_new_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=A_XYZ_DISPLAY_INDEX_NAME,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=150,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        index_update_method="STREAM_UPDATE"
    )
    
    if a_new_index:
        print(f"Created index: {a_new_index.name}")
    
    list_indexes = aiplatform.MatchingEngineIndex.list()
    print(f"Current indexes: {list_indexes}")
    
    # Step 7: Create new index endpoint
    print("\nStep 7: Creating new index endpoint")
    a_new_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=A_XYZ_DISPLAY_END_POINT_NAME,
        public_endpoint_enabled=True
    )
    
    if a_new_endpoint:
        print(f"Created endpoint: {a_new_endpoint.name}")
    
    list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()
    print(f"Current endpoints: {list_end_points}")
    
    # Step 8: Deploy index with public endpoint
    print("\nStep 8: Deploying index with public endpoint")
    a_new_endpoint = a_new_endpoint.deploy_index(
        index=a_new_index,
        deployed_index_id=A_XYZ_DEPLOYED_INDEX_ID
    )
    
    print(f"Deployed indexes: {a_new_endpoint.deployed_indexes}")
    print("\nPhase 4 completed successfully!")
    
    return a_new_index, a_new_endpoint

def phase_10(project_id, location="us-central1"):
    """
    Phase 10: Un-deploy indexes and delete all indexes & endpoints
    """
    print("Starting Phase 10: Un-deploy indexes and delete all indexes & endpoints")
    
    # Initialize Vertex AI
    initialize_vertex_ai(project_id, location)
    
    # Step 1: Verify existing endpoints
    print("\nStep 1: Verifying existing endpoints")
    list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()
    print(f"List of endpoints: {list_end_points}")
    
    # Step 2 & 3: Un-deploy and delete endpoints
    print("\nSteps 2 & 3: Un-deploying and deleting endpoints")
    for endpoint in list_end_points:
        print(f"Processing endpoint: {endpoint.name}")
        endpoint.undeploy_all()
        endpoint.delete()
    
    # Verify endpoints are deleted
    list_end_points = aiplatform.MatchingEngineIndexEndpoint.list()
    print(f"Remaining endpoints: {list_end_points}")
    
    # Step 4: Verify existing indexes
    print("\nStep 4: Verifying existing indexes")
    list_indexes = aiplatform.MatchingEngineIndex.list()
    print(f"List of indexes: {list_indexes}")
    
    # Step 5 & 6: Delete indexes
    print("\nSteps 5 & 6: Deleting indexes")
    for index in list_indexes:
        print(f"Deleting index: {index.name}")
        index.delete()
    
    # Verify indexes are deleted
    list_indexes = aiplatform.MatchingEngineIndex.list()
    print(f"Remaining indexes: {list_indexes}")
    
    print("\nPhase 10 completed successfully!")

# Example usage in a Jupyter Notebook:
"""
# Initialize with your project details
PROJECT_ID = "your-project-id"  # Replace with your actual project ID
LOCATION = "us-central1"  # Or your preferred location

# Run Phase 4
new_index, new_endpoint = phase_4(PROJECT_ID, LOCATION)

# Run Phase 10 (after completing the project)
# phase_10(PROJECT_ID, LOCATION)
""" 