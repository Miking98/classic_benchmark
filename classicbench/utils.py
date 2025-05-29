import argparse
import os
from typing import Dict, Optional
from omegaconf import DictConfig, OmegaConf
import hashlib
import pandas as pd

def load_config(args: argparse.Namespace) -> DictConfig:
    """Use OmegaConf to load YAML config files."""
    config_data = OmegaConf.load(args.data)
    config_agent = OmegaConf.load(args.agent)
    config_eval = OmegaConf.load(args.eval)
    config = OmegaConf.merge(config_data, config_agent, config_eval)
    return config

def flatten_dict(d: Dict, parent_key: str = '', separator: str = '-') -> Dict:
    """Flatten a nested dictionary. Merge nested keys with '-'. Modified from https://stackoverflow.com/a/6027615/119527"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def hashstring(text: str) -> str:
    """Hash a string"""
    # Hash the input string using SHA-256
    hash_object = hashlib.sha256(text.encode())
    # Get the hexadecimal digest and truncate it to 48 characters
    fixed_length_hash = hash_object.hexdigest()[:48]
    return fixed_length_hash

def seed_from_string(string: str) -> int:
    """Convert a string to a seed to use with `random` for reproducibility"""
    # Create a hash from the string and convert it to an integer
    seed = int(hashlib.sha256(str(string).encode('utf-8')).hexdigest(), 16) % (2**32)
    return seed

def override_config_with_cli_args(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Override args in args with CLI flags"""
    for arg in args:
        if arg.startswith("--"):
            # Remove the '--' prefix and split by '.'
            keys, value = arg[2:].split('=')
            keys = keys.split('.')
            
            # Convert value to appropriate type (int, float, bool, etc.)
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            # Traverse the keys and update the config
            for key in keys[:-1]:
                config = config[key]  # Traverse to the right section
            config[keys[-1]] = value  # Set the final key's value
    return config

def get_rel_path(path: str) -> str:
    """Get the relative path from the root of the project"""
    return os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')), path))

def map_annotator_workflow_id_to_true_workflow_id(annotator_workflow_id: str, dataset: str, path_to_workflow_mapping_csv: str = get_rel_path(f'data/annotations_support/workflow_mapping.csv')) -> Optional[str]:
    """Map annotator workflow id to true workflow id. 
    
        Example:
            annotator_workflow_id="5", dataset='hr' ==> returns "089f3327-6679-4eaf-91bc-51efbd3a290b" 
    """
    # Get mapping for this dataset
    df_labels = pd.read_csv(path_to_workflow_mapping_csv, dtype={'annotator_workflow_id': str, 'workflow_uuid': str, 'dataset' : str, })
    dataset = str(dataset.lower())
    assert dataset in df_labels['dataset'].unique(), f"Dataset {dataset} not found in workflow_mapping.csv"
    df_labels = df_labels[df_labels['dataset'] == dataset]
    
    # Handle None case
    if annotator_workflow_id in [None, 'None', 'none', '']:
        return None
    
    # Retrieve true workflow id
    annotator_workflow_id = str(annotator_workflow_id)
    assert annotator_workflow_id in df_labels['annotator_workflow_id'].unique(), f"Annotator workflow id {annotator_workflow_id} not found in workflow_mapping.csv"
    true_workflow_id = df_labels[df_labels['annotator_workflow_id'] == annotator_workflow_id]['workflow_uuid'].values[0]
    return true_workflow_id

if __name__ == '__main__':
    assert map_annotator_workflow_id_to_true_workflow_id('5', 'hr') == '089f3327-6679-4eaf-91bc-51efbd3a290b'
    assert map_annotator_workflow_id_to_true_workflow_id(108, 'finance') == 'fffa7661-3a02-cf0d-e2c9-c1ea05fa99d8'
    assert map_annotator_workflow_id_to_true_workflow_id(None, 'biotech') == None
    assert map_annotator_workflow_id_to_true_workflow_id('', 'finance') == None
    assert map_annotator_workflow_id_to_true_workflow_id(13, 'edtech') == 'e11d0604-6b47-041b-4710-557cbec71feb'