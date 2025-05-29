"""
Given a YAML config, load the corresponding dataset and dataloader.
"""

import os
import pickle
from typing import Dict
import pandas as pd
from classicbench.data.base import Dataloader, Dataset
from omegaconf import DictConfig
from datasets import load_dataset as load_hf_dataset

###################################################
###################################################
#
# Dataset Loaders
#
###################################################
###################################################

def load_dataloader(config: DictConfig, dataset: Dataset, is_use_cache: bool = True) -> Dataloader:
    """Create dataloader from dataset."""
    print(f"==> Loading dataloader from {config.dataloader}")
    path_to_cache_dir = config.dataloader.cache_dir
    os.makedirs(path_to_cache_dir, exist_ok=True)
    
    # Check if dataloader is cached, and load it
    dataset_uuid = dataset.get_uuid()
    if is_use_cache and os.path.exists(os.path.join(path_to_cache_dir, f"dataloader-{dataset_uuid}.pkl")):
        print("Loaded dataloader from cache.")
        dataloader = pickle.load(open(os.path.join(path_to_cache_dir, f"dataloader-{dataset_uuid}.pkl"), "rb"))
        return dataloader
    
    # Create dataloader from scratch
    print("Creating dataloader from scratch.")
    dataloader = Dataloader(config.dataloader, dataset)
    
    # Save dataloader
    pickle.dump(dataloader, open(os.path.join(path_to_cache_dir, f"dataloader-{dataset_uuid}.pkl"), "wb"))

    return dataloader

def load_dataset(config: DictConfig, is_print_log: bool = True, is_force_refresh: bool = False) -> Dataset:
    """Load dataset from CSV files."""
    if is_print_log:
        print(f"==> Loading dataset from {config.dataset}")

    # Load dataset subsets
    df_domains = load_hf_dataset(config.dataset.path_or_name, 'domains', download_mode='force_redownload' if is_force_refresh else None)['test'].to_pandas()
    df_workflows = load_hf_dataset(config.dataset.path_or_name, 'workflows', download_mode='force_redownload' if is_force_refresh else None)['test'].to_pandas()
    df_messages = load_hf_dataset(config.dataset.path_or_name, 'messages', download_mode='force_redownload' if is_force_refresh else None)['test'].to_pandas()
    df_jailbreakprompts = load_hf_dataset(config.dataset.path_or_name, 'jailbreak_prompts', download_mode='force_redownload' if is_force_refresh else None)['test'].to_pandas()
    
    # Replace nan with None
    df_messages = df_messages.replace({float('nan'): None})
    df_domains = df_domains.replace({float('nan'): None})
    df_workflows = df_workflows.replace({float('nan'): None})
    
    # Casting to str
    df_messages['conversation_uuid'] = df_messages['conversation_uuid'].astype('str')
    df_messages['true_workflow_uuid'] = df_messages['true_workflow_uuid'].astype('str')
    df_messages['true_workflow_uuid_2'] = df_messages['true_workflow_uuid_2'].astype('str')

    # Split df_messages into a separate message per row (i.e. request / response will be separate rows instead of one row with both request and response)
    df_messages = split_df_messages_into_row_per_message(df_messages)

    return Dataset(config.dataset, df_messages, df_domains, df_workflows, df_jailbreakprompts)

def split_df_messages_into_row_per_message(df_messages: pd.DataFrame) -> pd.DataFrame:
    """Splits df_messages into a separate message per row (i.e. request / response will be separate rows instead of one row with both request and response)."""
    # Data cleaning
    pd.set_option('future.no_silent_downcasting', True)
    df_messages['request_content'] = df_messages['request_content'].fillna('')
    df_messages['response_content'] = df_messages['response_content'].fillna('')

    # Split df_messages into a separate message for each row
    messages = []
    for idx, row in df_messages.iterrows():
        shared: Dict[str, str] = {
            'conversation_uuid': str(row['conversation_uuid']),
            'domain_uuid': row['domain_uuid'],
            'true_workflow_uuid': None if pd.isna(row['true_workflow_uuid']) else row['true_workflow_uuid'],
            'true_workflow_uuid_2': None if pd.isna(row['true_workflow_uuid_2']) else row['true_workflow_uuid_2'],
        }
        # request = user
        messages.append({
            'message_uuid': str(row['conversation_uuid']) + '--' + str(row['request_idx'] * 2),
            'message_idx': row['request_idx'] * 2,
            'sender' : 'user',
            'content': row['request_content'],
            **shared,
        })
        # response = bot
        if row['response_content'] != '':
            messages.append({
                'message_uuid': str(row['conversation_uuid']) + '--' + str(row['request_idx'] * 2 + 1),
                'message_idx': row['request_idx'] * 2 + 1,
                'sender' : 'bot',
                'content': row['response_content'],
                **shared,
            })
    df_messages = pd.DataFrame(messages)
    df_messages = df_messages.replace({float('nan'): None})
    return df_messages