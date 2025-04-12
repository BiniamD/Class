"""
Cache Manager for FinancialIQ
Handles caching of processed document results and embeddings
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

class CacheManager:
    """Manages caching of processed document results and embeddings"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.document_cache = os.path.join(cache_dir, "documents")
        self.embedding_cache = os.path.join(cache_dir, "embeddings")
        
        # Create cache directories if they don't exist
        os.makedirs(self.document_cache, exist_ok=True)
        os.makedirs(self.embedding_cache, exist_ok=True)
        
        # Cache expiration time (30 days)
        self.expiration_time = timedelta(days=30)

    def _get_cache_path(self, key: str) -> str:
        """Get the path for a cache file"""
        return os.path.join(self.cache_dir, f"{key}.json")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid"""
        if not os.path.exists(cache_path):
            return False
            
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < self.expiration_time

    def get_document_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached document processing results"""
        cache_path = self._get_cache_path(os.path.basename(file_path))
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def set_document_cache(self, file_path: str, data: Dict[str, Any]):
        """Cache document processing results"""
        cache_path = self._get_cache_path(os.path.basename(file_path))
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def get_embedding_cache(self, text: str) -> Optional[list]:
        """Get cached embedding for text"""
        cache_path = self._get_cache_path(text)
        
        if not self._is_cache_valid(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading embedding cache: {str(e)}")
            return None

    def set_embedding_cache(self, text: str, embedding: list) -> None:
        """Cache embedding for text"""
        cache_path = self._get_cache_path(text)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            print(f"Error writing embedding cache: {str(e)}")

    def clear_expired_cache(self) -> None:
        """Clear expired cache files"""
        for cache_type in ["document", "embedding"]:
            cache_dir = self.document_cache if cache_type == "document" else self.embedding_cache
            for filename in os.listdir(cache_dir):
                cache_path = os.path.join(cache_dir, filename)
                if not self._is_cache_valid(cache_path):
                    try:
                        os.remove(cache_path)
                    except Exception as e:
                        print(f"Error clearing cache: {str(e)}")

    def clear_all_cache(self) -> None:
        """Clear all cache files"""
        for cache_type in ["document", "embedding"]:
            cache_dir = self.document_cache if cache_type == "document" else self.embedding_cache
            for filename in os.listdir(cache_dir):
                try:
                    os.remove(os.path.join(cache_dir, filename))
                except Exception as e:
                    print(f"Error clearing cache: {str(e)}")

    def clear_cache(self):
        """Clear all cached data"""
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, file))

    def get_cache_size(self) -> int:
        """Get total size of cache in bytes"""
        total_size = 0
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                total_size += os.path.getsize(os.path.join(self.cache_dir, file))
        return total_size 