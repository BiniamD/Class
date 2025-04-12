"""
FinancialIQ: API Key Setup Script
Creates and configures a Google Cloud API key for the application
"""

import os
from datetime import datetime
from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key
from dotenv import load_dotenv

def create_api_key(project_id: str, suffix: str) -> Key:
    """
    Creates and restrict an API key. Add the suffix for uniqueness.
    
    Args:
        project_id: Google Cloud project id.
        suffix: Unique suffix for the API key name
        
    Returns:
        response: Returns the created API Key.
    """
    # Create the API Keys client
    client = api_keys_v2.ApiKeysClient()

    key = api_keys_v2.Key()
    key.display_name = f"FinancialIQ-API-Key-{suffix}"

    # Initialize request and set arguments
    request = api_keys_v2.CreateKeyRequest()
    request.parent = f"projects/{project_id}/locations/global"
    request.key = key

    # Make the request and wait for the operation to complete
    response = client.create_key(request=request).result()

    print(f"Successfully created an API key: {response.name}")
    return response

def update_env_file(api_key: str):
    """Updates the .env file with the new API key"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    
    with open(env_path, 'r') as file:
        lines = file.readlines()
    
    # Update or add the GOOGLE_API_KEY line
    api_key_updated = False
    for i, line in enumerate(lines):
        if line.startswith('GOOGLE_API_KEY='):
            lines[i] = f'GOOGLE_API_KEY={api_key}\n'
            api_key_updated = True
            break
    
    if not api_key_updated:
        lines.append(f'GOOGLE_API_KEY={api_key}\n')
    
    with open(env_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Updated .env file with new API key")

def main():
    # Load environment variables
    load_dotenv()
    
    # Get project ID from environment
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
    
    # Create a unique suffix using timestamp
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    try:
        # Create the API key
        response = create_api_key(project_id, suffix)
        
        # Update the .env file with the new key
        update_env_file(response.key_string)
        
        print("\nAPI key setup completed successfully!")
        print("The key has been saved to your .env file")
        print("\nIMPORTANT: Make sure to restrict this API key's usage in the Google Cloud Console")
        print(f"API Key resource name: {response.name}")
        
    except Exception as e:
        print(f"Error setting up API key: {str(e)}")
        raise

if __name__ == "__main__":
    main() 