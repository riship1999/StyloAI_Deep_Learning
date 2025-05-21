import os
import json
from datetime import datetime
import shutil
import hashlib
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersion:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.versions_file = os.path.join(model_dir, 'versions.json')
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict:
        """Load version history from versions.json"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {'models': {}, 'current': {}}
    
    def _save_versions(self):
        """Save version history to versions.json"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute SHA-256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_version(self, 
                    model_name: str, 
                    model_path: str, 
                    metrics: Dict[str, float],
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a new model version"""
        # Create version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_{timestamp}"
        
        # Create version directory
        version_dir = os.path.join(self.model_dir, 'versions', version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model file
        model_filename = os.path.basename(model_path)
        new_model_path = os.path.join(version_dir, model_filename)
        shutil.copy2(model_path, new_model_path)
        
        # Compute model hash
        model_hash = self._compute_model_hash(new_model_path)
        
        # Create version info
        version_info = {
            'id': version_id,
            'timestamp': timestamp,
            'metrics': metrics,
            'hash': model_hash,
            'path': new_model_path,
            'metadata': metadata or {}
        }
        
        # Update versions
        if model_name not in self.versions['models']:
            self.versions['models'][model_name] = {}
        self.versions['models'][model_name][version_id] = version_info
        
        # Save versions file
        self._save_versions()
        
        logger.info(f"Saved version {version_id} for model {model_name}")
        return version_id
    
    def set_current_version(self, model_name: str, version_id: str):
        """Set the current version for a model"""
        if model_name not in self.versions['models']:
            raise ValueError(f"Model {model_name} not found")
        if version_id not in self.versions['models'][model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        self.versions['current'][model_name] = version_id
        self._save_versions()
        
        logger.info(f"Set current version of {model_name} to {version_id}")
    
    def get_current_version(self, model_name: str) -> Dict:
        """Get the current version info for a model"""
        if model_name not in self.versions['current']:
            raise ValueError(f"No current version set for model {model_name}")
        
        version_id = self.versions['current'][model_name]
        return self.versions['models'][model_name][version_id]
    
    def list_versions(self, model_name: str) -> Dict:
        """List all versions for a model"""
        if model_name not in self.versions['models']:
            raise ValueError(f"Model {model_name} not found")
        return self.versions['models'][model_name]
    
    def compare_versions(self, model_name: str, version_id1: str, version_id2: str) -> Dict:
        """Compare metrics between two versions"""
        versions = self.versions['models'][model_name]
        if version_id1 not in versions or version_id2 not in versions:
            raise ValueError("Version not found")
        
        v1_metrics = versions[version_id1]['metrics']
        v2_metrics = versions[version_id2]['metrics']
        
        comparison = {}
        for metric in set(v1_metrics.keys()) | set(v2_metrics.keys()):
            v1_value = v1_metrics.get(metric)
            v2_value = v2_metrics.get(metric)
            if v1_value is not None and v2_value is not None:
                diff = v2_value - v1_value
                rel_diff = (diff / v1_value) * 100 if v1_value != 0 else float('inf')
                comparison[metric] = {
                    'v1': v1_value,
                    'v2': v2_value,
                    'absolute_diff': diff,
                    'relative_diff_percent': rel_diff
                }
        
        return comparison
