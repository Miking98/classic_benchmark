"""
Usage:
    python3 3_convert_annotations_to_clean.py

Purpose:
    Given a folder containing Excel files (each containing one AMT annotator's raw annotations), 
    merge them into a single dataframe and filter out low quality conversations.
    
    Expects a folder containing multiple .xlsx files, each .xlsx file with the same structure but from a different annotator.
    Outputs multiple .csv files, one per dataset, with the filtered conversations.
"""
import base64
import collections
from typing import Dict, List
import uuid
import pandas as pd
import argparse
import os
from classicbench.utils import get_rel_path
from classicbench.private.aisera_metrics import calc_aisera_cost, calc_aisera_latency, calc_aisera_tokens

DATASET_NAMES: List[str] = [ x for x in os.listdir(get_rel_path("data/1_sampled")) if os.path.isdir(get_rel_path(f"data/1_sampled/{x}")) ]
BOOL_COLS: List[str] = ['Is gibberish?', 'Is asking for live human agent / call representative?', 'Is profanity?', 'Is PII? (e.g phone number, email, etc.)', 'Is bot response contain list of options?']
OPTIONAL_INT_COLS: List[str] = ['Workflow ID (1)', 'Workflow ID (2)']
MAJORITY_VOTE_COLS: List[str] = BOOL_COLS + OPTIONAL_INT_COLS

def parse_args():
    parser = argparse.ArgumentParser(description="Clean annotated dataset")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/2_annotations"), help="Path to the dataset directory. This should contain multiple .xlsx files, each .xlsx file with the same structure but from a different annotator.")
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("data/3_clean"), help="Path to the output directory. This will contain the filtered .csv files.")
    return parser.parse_args()

#################################
# Load / Aggregate Annotations
#################################
def majority_vote(series):
    """Returns the most common value in the series"""
    return collections.Counter(series).most_common(1)[0][0]

def aggregate_annotations(dfs: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Aggregates annotations across annotators into one dataframe per dataset"""
    for dataset_name, df_list in dfs.items():
        print(f"Majority voting within {dataset_name}...")
        df: pd.DataFrame = pd.concat(df_list)

        # Segment columns by voting type
        majority_cols = [col for col in df.columns if col in MAJORITY_VOTE_COLS]
        first_cols = [col for col in df.columns if col not in majority_cols]
        
        # Do majority vote across all annotators
        df = df.groupby('row_idx').agg({
            'row_idx': 'first',
            **{col: majority_vote for col in majority_cols},
            **{col: 'first' for col in first_cols},
        })
        dfs[dataset_name] = df
        print(f"Done aggregating within {dataset_name}. Shape: {df.shape}")
    return dfs

def load_annotations(path_to_dataset_dir: str) -> Dict[str, List[pd.DataFrame]]:
    """Loads annotations from the dataset directory"""
    dfs: Dict[str, List[pd.DataFrame]] = collections.defaultdict(list) # [key] = dataset name, [value] = dataframe
    for file in os.listdir(path_to_dataset_dir):
        if not file.endswith('.xlsx') or file.startswith('~'): # NOTE: ~ ignores temporary files created by Excel when file is currently open
            continue
        path_to_dataset_file: str = os.path.join(path_to_dataset_dir, file)
        for dataset_name in DATASET_NAMES:
            # check that sheet name exists
            if dataset_name not in pd.ExcelFile(path_to_dataset_file).sheet_names:
                continue
                
            # load sheet
            df = pd.read_excel(path_to_dataset_file, sheet_name=dataset_name, dtype={
                'Workflow ID (1)': str,
                'Workflow ID (2)': str,
                'triggering': str,
            })
            df['row_idx'] = df.index
            
            if 'triggering' in df.columns:
                # Cast "N/A" in `triggering` to None
                df['triggering'] = df['triggering'].apply(lambda x: x if str(x) != 'N/A' else None)
                # base64 decode `triggering` column
                df['triggering'] = df['triggering'].apply(lambda x: base64.b64decode(str(x)).decode('utf-8') if (x is not None and not pd.isna(x)) else None)

            # Cast Yes/No columns to boolean
            for col in BOOL_COLS:
                assert col in df.columns, f"Column {col} not found in {dataset_name}"
                df[col] = df[col].apply(lambda x: x == 'Yes')
            # Cast optional integer columns to string (b/c could be empty)
            for col in OPTIONAL_INT_COLS:
                assert col in df.columns, f"Column {col} not found in {dataset_name}"
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x != 'nan' else None)
            
            # Rename columns
            df.rename(columns={
                'Conversation ID': 'conversation_uuid',
                'User Message': 'request_content',
                'Bot Response': 'response_content',
                'triggering' : 'aisera_workflow_uuid',
                'Workflow ID (1)' : 'true_workflow_uuid',
                'Workflow ID (2)' : 'true_workflow_uuid_2',
                'Is gibberish?': 'is_gibberish',
                'Is asking for live human agent / call representative?': 'is_live_agent',
                'Is profanity?': 'is_profanity',
                'Is PII? (e.g phone number, email, etc.)': 'is_pii',
                'Is bot response contain list of options?': 'is_bot_options',
            }, inplace=True)
            
            df['request_content'] = df['request_content'].astype(str)
            df['response_content'] = df['response_content'].astype(str)
            df['domain_uuid'] = dataset_name.lower()

            dfs[dataset_name].append(df)
    return dfs

def load_workflows(path_to_dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads workflows from the dataset directory. Workflows are the same across all annotators, so no need to aggregate."""
    dfs: Dict[str, pd.DataFrame] = {}
    for file in os.listdir(path_to_dataset_dir):
        if not file.endswith('.xlsx') or file.startswith('~'): # NOTE: ~ ignores temporary files created by Excel when file is currently open
            continue
        path_to_dataset_file: str = os.path.join(path_to_dataset_dir, file)
        for dataset_name in DATASET_NAMES:
            df = pd.read_excel(path_to_dataset_file, sheet_name=f'{dataset_name}_workflows', dtype={
                'ID': str,
            })
            # Rename columns to match the cleaned dataset
            df.rename(columns={
                'ID': 'workflow_uuid',
                'Name': 'name',
                'Description': 'description',
            }, inplace=True)
            df = df[['workflow_uuid', 'name', 'description', ]]
            df['workflow_uuid'] = df['workflow_uuid'].astype(str).apply(lambda x: x if x != 'nan' else None)
            df['domain_uuid'] = dataset_name.lower()
            dfs[dataset_name] = df
        break # Only need to load one file
    return dfs

#################################
# Filters
#################################
def drop_gibberish(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => 'Is gibberish?' is True"""
    invalid_rows = df[df['is_gibberish'] == True]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]

def drop_profanity(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => 'Is profanity?' is True"""
    invalid_rows = df[df['is_profanity'] == True]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]

def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => 'Is PII? (e.g phone number, email, etc.)' is True"""
    invalid_rows = df[df['is_pii'] == True]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]

def drop_live_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => 'Is asking for live human agent / call representative?' is True"""
    invalid_rows = df[df['is_live_agent'] == True]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]

def drop_lingering_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with PII that we didn't catch originally in dataset generation process."""
    # For `medical` domain, drop rows where `request_content` contains "Ultra"
    invalid_rows = df[(df['domain_uuid'] == 'medical') & [ 'ultra' in x.lower() for x in df['request_content'] ]]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]
    
def drop_response_contains_list_of_options(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => the bot response contains a list of options.
        ! NOTE: Currently, this filter is not used.
    """
    invalid_rows = df[df['is_bot_options'] == True]
    return df[~df['conversation_uuid'].isin(invalid_rows['conversation_uuid'])]

def main():
    args = parse_args()

    # Load annotations -- each .xlsx file in directory is from a different annotator
    dfs_annotations: Dict[str, List[pd.DataFrame]] = load_annotations(args.path_to_dataset_dir)
    print(f"Loaded {len(dfs_annotations)} datasets with {sum(len(df_list) for df_list in dfs_annotations.values())} total annotators")

    # Load workflows (uniform across all annotators, so no need to aggregate)
    dfs_workflows = load_workflows(args.path_to_dataset_dir)
    print(f"Loaded {len(dfs_workflows)} sets of workflows")
    
    # Save domains
    dfs_domains = {} # [key] = dataset name, [value] = dataframe

    # Aggregate annotations across annotators => one dataframe per dataset
    dfs_annotations = aggregate_annotations(dfs_annotations)
    dfs_messages = {} # [key] = dataset name, [value] = dataframe

    # Filter out low quality conversations
    drop_stats = collections.defaultdict(dict) # [key] = domain, [value] = dict of drop stats
    total_start_n_rows = 0
    for domain, df in dfs_annotations.items():
        start_n_rows = df.shape[0]
        print("-" * 100)
        print(f"Filtering {domain}")
        print(f"=> Start | n_rows={start_n_rows}")
        print(f"  Filter{'':<47} | dropped_rows")
        print("  " + "-" * 70)
        for filter_fn in [drop_gibberish, drop_profanity, drop_pii, drop_live_agent, drop_lingering_pii]:
            old_n_rows = df.shape[0]
            df = filter_fn(df)
            drop_stats[domain][filter_fn.__name__] = old_n_rows - df.shape[0]
            print(f"   {filter_fn.__name__:<50} | {old_n_rows - df.shape[0]:>5}")
        print(f"=> Finished | n_rows={df.shape[0]} | dropped_rows={start_n_rows - df.shape[0]} ({(start_n_rows - df.shape[0]) / start_n_rows * 100:.2f}%)")
        total_start_n_rows += start_n_rows
        dfs_annotations[domain] = df
    
    # Print overall drop stats
    TABLE_LENGTH = 120
    FILTER_FNS: List[str] = list(drop_stats[list(drop_stats.keys())[0]].keys())
    print("-" * TABLE_LENGTH)
    print("Overall drop stats:")
    print("  " + "-" * TABLE_LENGTH)
    print(f"  {'Domain':<10} |", end="")
    for filter_fn in FILTER_FNS:
        print(f"  {filter_fn:<5} |", end="")
    print()
    print("-" * TABLE_LENGTH)
    for domain, domain_drop_stats in drop_stats.items():
        print(f"  {domain:<10} |", end="")
        for filter_fn in FILTER_FNS:
            print(f"  {domain_drop_stats[filter_fn]:<5} |", end="")
        print()
    # Print total drop stats
    total_drop_stats = {}
    for domain, domain_drop_stats in drop_stats.items():
        for filter_fn in FILTER_FNS:
            total_drop_stats[filter_fn] = total_drop_stats.get(filter_fn, 0) + domain_drop_stats[filter_fn]
    print("  " + "=" * TABLE_LENGTH)
    print(f"  {'Total':<10} |", end="")
    for filter_fn in FILTER_FNS:
        print(f"  {total_drop_stats[filter_fn]:<5} |", end="")
    print()
    print("  " + "-" * TABLE_LENGTH)
    print(f"Total starting rows: {total_start_n_rows}")
    print(f"Total dropped rows: {sum(total_drop_stats.values())}")
    print(f"Total final rows: {total_start_n_rows - sum(total_drop_stats.values())}")
    
    # Calculate Aisera token counts, costs, and latency
    for domain, df in dfs_annotations.items():
        df['aisera_n_tokens'] = calc_aisera_tokens(df)
        df['aisera_latency'] = calc_aisera_latency(df)
        df['aisera_cost'] = calc_aisera_cost(df)
    
    # Save filtered dataframes
    for domain, df in dfs_annotations.items():
        # Create output directory
        path_to_output_dir: str = os.path.join(args.path_to_output_dir, domain.lower())
        os.makedirs(path_to_output_dir, exist_ok=True)

        # Save messages.csv
        df = df[['conversation_uuid', 'request_content', 'response_content', 'true_workflow_uuid', 'true_workflow_uuid_2', 'aisera_workflow_uuid', 'aisera_n_tokens', 'aisera_cost', 'aisera_latency',]].copy()
        df['request_idx'] = df.groupby('conversation_uuid').cumcount()
        df['domain_uuid'] = domain.lower()
        df.to_csv(os.path.join(path_to_output_dir, f"messages.csv"), index=False)
        dfs_messages[domain] = df

        # Save domains.csv
        df_domains = pd.DataFrame({
            'domain_uuid': [ domain.lower() ],
            'domain_name': [ domain ],
        })
        df_domains.to_csv(os.path.join(path_to_output_dir, f"domains.csv"), index=False)
        dfs_domains[domain] = df_domains

        # Save workflows.csv
        dfs_workflows[domain].to_csv(os.path.join(path_to_output_dir, f"workflows.csv"), index=False)

    # Merge all CSVs into one directory
    df_messages = pd.concat([df for df in dfs_messages.values()])
    df_domains = pd.concat([df for df in dfs_domains.values()])
    df_workflows = pd.concat([df for df in dfs_workflows.values()])
    df_workflows.drop_duplicates(subset=['workflow_uuid'], inplace=True)

    # Save merged CSVs
    df_messages.to_csv(os.path.join(args.path_to_output_dir, f"messages.csv"), index=False)
    df_domains.to_csv(os.path.join(args.path_to_output_dir, f"domains.csv"), index=False)
    df_workflows.to_csv(os.path.join(args.path_to_output_dir, f"workflows.csv"), index=False)

if __name__ == "__main__":
    main()
