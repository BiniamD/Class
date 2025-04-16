"""
Script to set up and configure Vertex AI Matching Engine for FinancialIQ.
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Added for loading .env file
from dotenv import load_dotenv, set_key

import google.cloud.aiplatform as aiplatform
from google.cloud import storage
from langchain_community.vectorstores import MatchingEngine
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VertexAISetup:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION")
        self.a_new_index = None
        self.embeddings = VertexAIEmbeddings(
            model_name=os.getenv("VERTEX_AI_EMBEDDING_MODEL"),
            project=self.project_id,
            location=self.location
        )
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        
    def cleanup_existing_resources(self, index_name: str, endpoint_name: str) -> None:
        """Clean up existing Vertex AI resources."""
        try:
            # Check and delete existing index
            try:
                # List all indexes and find the one with matching display name
                indexes = aiplatform.MatchingEngineIndex.list()
                for index in indexes:
                    if index.display_name == index_name:
                        logger.info(f"Found existing index {index_name} with ID {index.name}, deleting...")
                        index.delete(force=True)
                        logger.info(f"Deleted existing index {index_name}")
                        break
                else:
                    logger.info(f"No existing index found with name {index_name}")
            except Exception as e:
                logger.info(f"Error checking/deleting index: {str(e)}")

            # Check and delete existing endpoint
            try:
                # List all endpoints and find the one with matching display name
                endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
                for endpoint in endpoints:
                    if endpoint.display_name == endpoint_name:
                        logger.info(f"Found existing endpoint {endpoint_name} with ID {endpoint.name}, deleting...")
                        endpoint.delete(force=True)
                        logger.info(f"Deleted existing endpoint {endpoint_name}")
                        break
                else:
                    logger.info(f"No existing endpoint found with name {endpoint_name}")
            except Exception as e:
                logger.info(f"Error checking/deleting endpoint: {str(e)}")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
            
    def create_gcs_bucket(self, bucket_name: str) -> None:
        """Create a GCS bucket if it doesn't exist."""
        storage_client = storage.Client(project=self.project_id)
        try:
            # Check if bucket exists
            bucket = storage_client.bucket(bucket_name)
            if bucket.exists():
                logger.info(f"Bucket {bucket_name} already exists")
                return
                
            bucket = storage_client.create_bucket(bucket_name, location=self.location)
            logger.info(f"Created bucket {bucket_name} in {self.location}")
        except Exception as e:
            logger.error(f"Error creating bucket: {str(e)}")
            raise
            
    def create_matching_engine_index(self, 
                                   index_name: str,
                                   gcs_bucket_name: str) -> str:
        """Create a Matching Engine index using Tree-AH algorithm."""
        if not gcs_bucket_name:
            raise ValueError("GCS bucket name must be provided.")
            
        try:
            # Check if index already exists
            try:
                index = aiplatform.MatchingEngineIndex(index_name=index_name)
                logger.info(f"Index {index_name} already exists with ID {index.name}")
                return index.name
            except Exception:
                logger.info(f"Creating new index {index_name}...")
            
            # Create new index
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=index_name,
                description="FinancialIQ SEC Filing Index",
                contents_delta_uri=f"gs://{gcs_bucket_name}/embeddings",
                dimensions=int(os.getenv("VERTEX_AI_DIMENSIONS")),
                approximate_neighbors_count=int(os.getenv("VERTEX_AI_APPROXIMATE_NEIGHBORS")),
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                leaf_node_embedding_count=int(os.getenv("VERTEX_AI_LEAF_NODE_EMBEDDING_COUNT")),
                leaf_nodes_to_search_percent=int(os.getenv("VERTEX_AI_LEAF_NODES_TO_SEARCH_PERCENT"))
            )
            self.a_new_index = index
            logger.info(f"Created Tree-AH index {index_name} with ID {index.name}")
            return index.name
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
            
    def create_index_endpoint(self, project_id: str, location: str, display_name: str) -> str:
        """Creates an index endpoint if it doesn't exist."""
        try:
            # Initialize the Vertex AI client
            client = aiplatform.gapic.IndexEndpointServiceClient(
                client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            )
            
            # Check if endpoint already exists
            parent = f"projects/{project_id}/locations/{location}"
            list_request = aiplatform.gapic.ListIndexEndpointsRequest(parent=parent)
            list_response = client.list_index_endpoints(request=list_request)
            
            for endpoint in list_response:
                if endpoint.display_name == display_name:
                    logger.info(f"Found existing endpoint: {endpoint.name}")
                    return endpoint.name
            
            # Create new endpoint if it doesn't exist
            endpoint = {
                "display_name": display_name,
                "description": "Endpoint for FinancialIQ vector search",
                "public_endpoint_enabled": True
            }
            
            create_request = aiplatform.gapic.CreateIndexEndpointRequest(
                parent=parent,
                index_endpoint=endpoint
            )
            
            operation = client.create_index_endpoint(request=create_request)
            response = operation.result()
            logger.info(f"Created new endpoint: {response.name}")
            return response.name
            
        except Exception as e:
            logger.error(f"Error creating index endpoint: {str(e)}")
            raise
            
    def deploy_index_to_endpoint(self, index_id: str, endpoint_id: str) -> None:
        """Deploy an index to an endpoint."""
        try:
            # Get the actual index and endpoint objects
            index = aiplatform.MatchingEngineIndex(index_id)
            endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_id)
            
            # Deploy the index
            endpoint.deploy_index(
                index=index,
                deployed_index_id=f"deployed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Deployed index {index_id} to endpoint {endpoint_id}")
        except Exception as e:
            logger.error(f"Error deploying index: {str(e)}")
            raise
            
    def process_and_upload_embeddings(self, documents: List[Document]) -> None:
        """Process documents into embeddings and upload to GCS."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP"))
        )
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings and format for upload
        embeddings_data = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            embeddings_data.append({
                "id": f"chunk_{i}",
                "embedding": embedding
            })
            
        # Upload to GCS
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(os.getenv("EMBEDDINGS_BUCKET"))
        blob = bucket.blob("embeddings/embeddings.jsonl")
        
        # Convert to JSON Lines format
        jsonl_content = "\n".join([json.dumps(item) for item in embeddings_data])
        blob.upload_from_string(jsonl_content)
        logger.info(f"Uploaded {len(embeddings_data)} embeddings to GCS")

    def cleanup_all_resources(self) -> None:
        """Clean up all Vertex AI resources (indexes and endpoints)."""
        try:
            # Step 1: List and cleanup endpoints
            logger.info("Step 1: Listing and cleaning up endpoints...")
            endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            if endpoints:
                logger.info(f"Found {len(endpoints)} endpoints to clean up")
                for endpoint in endpoints:
                    try:
                        logger.info(f"Processing endpoint: {endpoint.display_name} ({endpoint.name})")
                        # Undeploy all indexes from the endpoint
                        endpoint.undeploy_all()
                        logger.info(f"Undeployed all indexes from endpoint: {endpoint.display_name}")
                        # Delete the endpoint
                        endpoint.delete()
                        logger.info(f"Deleted endpoint: {endpoint.display_name}")
                    except Exception as e:
                        logger.error(f"Error processing endpoint {endpoint.display_name}: {str(e)}")
            else:
                logger.info("No endpoints found to clean up")

            # Step 2: List and cleanup indexes
            logger.info("Step 2: Listing and cleaning up indexes...")
            indexes = aiplatform.MatchingEngineIndex.list()
            if indexes:
                logger.info(f"Found {len(indexes)} indexes to clean up")
                for index in indexes:
                    try:
                        logger.info(f"Processing index: {index.display_name} ({index.name})")
                        # Delete the index
                        index.delete()
                        logger.info(f"Deleted index: {index.display_name}")
                    except Exception as e:
                        logger.error(f"Error processing index {index.display_name}: {str(e)}")
            else:
                logger.info("No indexes found to clean up")

            # Step 3: Verify cleanup
            logger.info("Step 3: Verifying cleanup...")
            remaining_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            remaining_indexes = aiplatform.MatchingEngineIndex.list()
            
            if not remaining_endpoints and not remaining_indexes:
                logger.info("✅ Cleanup successful: No remaining endpoints or indexes")
            else:
                if remaining_endpoints:
                    logger.warning(f"⚠️ {len(remaining_endpoints)} endpoints still exist")
                if remaining_indexes:
                    logger.warning(f"⚠️ {len(remaining_indexes)} indexes still exist")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

def update_env_file(env_path: str, updates: Dict[str, str]) -> None:
    """Update the .env file with new values."""
    try:
        # Load existing .env file
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Create a dictionary of existing variables
        existing_vars = {}
        for line in lines:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                existing_vars[key] = value
        
        # Update with new values
        existing_vars.update(updates)
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Updated .env file with new values")
    except Exception as e:
        logger.error(f"Error updating .env file: {str(e)}")
        raise

def main():
    """Main function to set up Vertex AI resources."""
    try:
        setup = VertexAISetup()
        
        # Get resource names from environment
        index_name = os.getenv("VERTEX_AI_INDEX_NAME")
        endpoint_name = os.getenv("VERTEX_AI_ENDPOINT_NAME")
        bucket_name = os.getenv("EMBEDDINGS_BUCKET")
        
        if not all([index_name, endpoint_name, bucket_name]):
            raise ValueError("Missing required environment variables for Vertex AI setup")
        
        # First, clean up all existing resources
        logger.info("Starting cleanup of all existing resources...")
        setup.cleanup_all_resources()
        
        logger.info("Starting Vertex AI setup...")
        logger.info(f"Index Name: {index_name}")
        logger.info(f"Endpoint Name: {endpoint_name}")
        logger.info(f"Bucket Name: {bucket_name}")
        
        # Create GCS bucket
        logger.info("Creating GCS bucket...")
        setup.create_gcs_bucket(bucket_name)
        
        # Create index and endpoint
        logger.info("Creating Vertex AI resources...")
        index_id = setup.create_matching_engine_index(index_name, bucket_name)
        endpoint_id = setup.create_index_endpoint(setup.project_id, setup.location, endpoint_name)
        
        # Deploy the index to the endpoint
        logger.info("Deploying index to endpoint...")
        setup.deploy_index_to_endpoint(index_id, endpoint_id)
        
        # Update .env file with resource IDs
        env_updates = {
            "VERTEX_AI_INDEX_ID": index_id,
            "VERTEX_AI_ENDPOINT_ID": endpoint_id,
            "VERTEX_AI_EMBEDDINGS_BUCKET": bucket_name
        }
        update_env_file(".env", env_updates)
        
        logger.info("Vertex AI setup completed successfully!")
        logger.info(f"Index Resource Name: {index_id}")
        logger.info(f"Endpoint Resource Name: {endpoint_id}")
        logger.info(f"GCS Bucket: {bucket_name}")
        logger.info("Index has been deployed to the endpoint.")
        logger.warning("Run the processing pipeline to generate embeddings and upload them to the GCS bucket.")
        logger.warning(f"Embeddings should be uploaded to: gs://{bucket_name}/embeddings/embeddings.jsonl")
        
    except Exception as e:
        logger.error(f"Error in Vertex AI setup: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 