import os
import json
import time
import traceback
from google.cloud import storage
from google.cloud import aiplatform
from google.auth import default, exceptions
from dotenv import load_dotenv

def load_embeddings_from_gcs(bucket_name: str, embeddings_path: str, storage_client) -> list:
    """Loads embeddings from a JSON Lines file in GCS."""
    print(f"Loading embeddings from gs://{bucket_name}/{embeddings_path}...")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(embeddings_path)

    if not blob.exists():
        print(f"❌ Error: Embeddings file not found at gs://{bucket_name}/{embeddings_path}")
        return []

    embeddings_data = []
    try:
        content = blob.download_as_text()
        for line in content.splitlines():
            if line.strip():
                try:
                    embeddings_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ Warning: Skipping invalid JSON line: {e}")
        print(f"✅ Successfully loaded {len(embeddings_data)} embeddings.")
        return embeddings_data
    except Exception as e:
        print(f"❌ Error downloading or parsing embeddings file: {e}")
        traceback.print_exc()
        return []

def index_embeddings(project_id: str, location: str, index_endpoint_name: str, embeddings_data: list, batch_size: int = 100):
    """Upserts embeddings to a Vertex AI Matching Engine Index."""
    if not embeddings_data:
        print("❌ No embeddings data to index.")
        return

    print(f"Initializing AI Platform client for project={project_id}, location={location}...")
    try:
        # Use explicitly set credentials if available, otherwise use ADC
        credentials, _ = default()
        aiplatform.init(project=project_id, location=location, credentials=credentials)
        print("✅ AI Platform client initialized.")
    except exceptions.DefaultCredentialsError as e:
         print(f"❌ Error initializing AI Platform: Could not find default credentials. {e}")
         print("   Please ensure you have authenticated via 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
         return
    except Exception as e:
        print(f"❌ Unexpected error initializing AI Platform: {e}")
        traceback.print_exc()
        return

    # Get the Index Endpoint
    print(f"Fetching Index Endpoint: {index_endpoint_name}...")
    try:
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'displayName="{index_endpoint_name}"',
            project=project_id,
            location=location,
        )[0] # Assuming unique display name
        print(f"✅ Found Index Endpoint: {index_endpoint.resource_name}")
    except IndexError:
        print(f"❌ Error: Index Endpoint '{index_endpoint_name}' not found in {location}.")
        return
    except Exception as e:
        print(f"❌ Error fetching Index Endpoint: {e}")
        traceback.print_exc()
        return

    # Prepare datapoints for upsert (list of IndexDatapoint objects)
    datapoints_to_upsert = [
        aiplatform.IndexDatapoint(
            datapoint_id=item["id"],
            feature_vector=item["embedding"]
            # Metadata can't be directly upserted this way, it's part of the index config/structure
        ) for item in embeddings_data if "id" in item and "embedding" in item
    ]

    if not datapoints_to_upsert:
        print("❌ No valid datapoints found in the loaded data.")
        return

    total_datapoints = len(datapoints_to_upsert)
    print(f"Prepared {total_datapoints} datapoints for upserting.")

    # Upsert in batches
    print(f"Upserting datapoints in batches of {batch_size}...")
    start_time = time.time()
    upserted_count = 0
    for i in range(0, total_datapoints, batch_size):
        batch = datapoints_to_upsert[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_datapoints + batch_size - 1) // batch_size
        print(f"  Upserting batch {batch_num}/{total_batches} ({len(batch)} datapoints)...")
        try:
            # Note: The upsert operation might take time, especially for the first few batches
            # as the index might be scaling or updating.
            index_endpoint.upsert_datapoints(datapoints=batch)
            upserted_count += len(batch)
            print(f"  ✅ Batch {batch_num} upserted successfully.")
            # Optional: Add a small delay between batches if needed, e.g., time.sleep(1)
        except Exception as e:
            print(f"  ❌ Error upserting batch {batch_num}: {e}")
            # Decide if you want to stop or continue with the next batch
            # For now, we'll stop on the first error.
            traceback.print_exc()
            break

    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Indexing Summary ---")
    print(f"Attempted to upsert {total_datapoints} datapoints.")
    print(f"Successfully upserted {upserted_count} datapoints.")
    print(f"Total time taken: {duration:.2f} seconds.")

if __name__ == "__main__":
    load_dotenv()

    # --- Configuration ---
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION")
    embeddings_gcs_bucket = os.getenv("EMBEDDINGS_BUCKET")
    vertex_index_endpoint_name = os.getenv("VERTEX_AI_ENDPOINT_NAME") # Use the Endpoint name
    gcs_embeddings_path = "embeddings/embeddings.jsonl" # Path within the bucket

    # Validate configuration
    if not all([gcp_project, gcp_location, embeddings_gcs_bucket, vertex_index_endpoint_name]):
        print("❌ Error: Missing required environment variables.")
        print("   Ensure GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, EMBEDDINGS_BUCKET, and VERTEX_AI_ENDPOINT_NAME are set in .env")
        exit(1)

    # --- Credential Check ---
    # Explicitly check if GOOGLE_APPLICATION_CREDENTIALS is set
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    storage_client = None
    if key_path:
        try:
            storage_client = storage.Client.from_service_account_json(key_path)
            print(f"🔑 Using service account key: {key_path}")
        except FileNotFoundError:
            print(f"❌ Error: Service account key file not found at '{key_path}'")
            exit(1)
        except Exception as e:
            print(f"❌ Error initializing storage client with key file: {e}")
            exit(1)
    else:
        try:
            # Try using Application Default Credentials
            storage_client = storage.Client(project=gcp_project)
            print("🔑 Using Application Default Credentials.")
        except exceptions.DefaultCredentialsError as e:
            print(f"❌ Error: Could not find default credentials. {e}")
            print("   Please ensure you have authenticated via 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
            exit(1)
        except Exception as e:
             print(f"❌ Error initializing storage client with ADC: {e}")
             exit(1)

    # --- Execution ---
    print("\n--- Starting Embedding Indexing ---")
    loaded_data = load_embeddings_from_gcs(embeddings_gcs_bucket, gcs_embeddings_path, storage_client)

    if loaded_data:
        index_embeddings(
            project_id=gcp_project,
            location=gcp_location,
            index_endpoint_name=vertex_index_endpoint_name,
            embeddings_data=loaded_data,
            batch_size=100 # Adjust batch size if needed (100 is a reasonable default)
        )

    print("\n--- Indexing Script Finished ---")