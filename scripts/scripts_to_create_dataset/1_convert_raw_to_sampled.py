"""Usage:
    python3 1_convert_raw_to_sampled.py

Purpose:
    Sample chats from the raw data dump.
    NOTE: Takes ~1 hr for 7 domains (mostly due to profanity filtering)
"""
import argparse
import collections
import json
import os
import random
import re
import time
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from better_profanity import profanity
from classicbench.utils import get_rel_path
from classicbench.private.deidentify_dataset import deidentify_text
from tabulate import tabulate
load_dotenv()

profanity.load_censor_words()

N_MESSAGES_LONG_CONVO_THRESHOLD: int = 2

DF_MESSAGE_COLUMNS = [
    'id',
    'conversation_uuid',
    'depth_of_conversation',
    'request_uuid',
    'request_idx',
    'request_content',
    'response_content',
    'triggering_workflow_uuid',
    'true_workflow_uuid',
    'true_workflow_uuid_2',
    'domain_uuid',
    'aisera_workflow_uuid',
    'aisera_n_tokens',
    'aisera_cost',
    'aisera_latency',
]

def parse_args():
    parser = argparse.ArgumentParser(description="Sample and Evaluate Chat Data")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/0_raw"), help="Path to the raw dataset dump from Aisera.")
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("data/1_sampled"), help="Path to where sampled chats will be saved.")
    parser.add_argument("--n_samples", type=int, default=1000, help="# of samples per domain to take.")
    return parser.parse_args()

def drop_messages_after_condition(df: pd.DataFrame, condition_mask: pd.Series) -> pd.DataFrame:
    """Helper function to drop all messages that occur after a condition is met within a conversation"""
    # Find the first request index where the condition is True for each conversation.
    first_invalid = (
        df.loc[condition_mask, ['conversation_uuid', 'request_idx']]
        .groupby('conversation_uuid', as_index=False)
        .min()
        .rename(columns={'request_idx': 'first_invalid_idx'})
    )
    
    # Merge the first invalid indices back into the original DataFrame.
    df_merged = df.merge(first_invalid, on='conversation_uuid', how='left')
    
    # Keep rows where either there's no invalid condition (NaN) or the request_idx is less than the invalid index.
    keep_mask = df_merged['first_invalid_idx'].isna() | (df_merged['request_idx'] < df_merged['first_invalid_idx'])
    
    return df_merged.loc[keep_mask, df.columns]

def drop_redacted_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop conversations where >=50% of messages contain 'REDACTED'"""
    conversation_uuid_2_messages: Dict[str, List[str]] = collections.defaultdict(list)
    for idx, row in df.iterrows():
        conversation_uuid_2_messages[row['conversation_uuid']].append(row['request_content'])
    drop_conversation_uuids: List[str] = []
    for conversation_uuid, messages in conversation_uuid_2_messages.items():
        if sum(1 for message in messages if 'REDACTED' in message) >= len(messages) * 0.5:
            drop_conversation_uuids.append(conversation_uuid)
    df = df[~df['conversation_uuid'].isin(drop_conversation_uuids)]
    return df

def drop_null_conversation_uuid(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where conversation_uuid is empty"""
    return drop_messages_after_condition(df, df['conversation_uuid'].isna() | (df['conversation_uuid'] == '') )

def drop_short_char_len(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => the length of the user message is <= 4 characters"""
    return drop_messages_after_condition(df, df['request_content'].str.len() <= 4)

def drop_cancel_or_exit(df: pd.DataFrame) -> pd.DataFrame:
    """Drops convos where at least one row => the user message is simply 'cancel' or 'exit'"""
    strings = [
        'cancel',
        'exit',
    ]
    return drop_messages_after_condition(df, df['request_content'].str.lower().isin(strings))

def drop_clear_button_presses(df: pd.DataFrame) -> pd.DataFrame:
    """Remove clear button presses -- e.g. Account Management, Direct Deposit, Check Direct Deposit Status."""
    # Drop convos marked as clicks
    invalid_uuids = df.loc[
        (df['conversation_request_type'] == 'UserClick') | 
        (df['conversation_response_producer'] == 'LiveAgent'), 
        'conversation_uuid'
    ].unique()
    df = df[~df['conversation_uuid'].isin(invalid_uuids)]
    # Drop convos that are probably clicks
    button_press_texts = [
        "Account Management",
        "Direct Deposit",
        "Check Direct Deposit Status",
    ]
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: any(text.lower() == x.lower() for text in button_press_texts)))

def drop_testing_chats(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are simply testing the chatbot, e.g. 'testing by aisera' or 'Testing_cancel_ticket'"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: ('testing' in x.lower() and 'aisera' in x.lower()) or ('testing_' in x.lower())))

def drop_language_preferences(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are simply language preferences, e.g. 'It looks like your language preference is not set'"""
    strings = [
        'It looks like your language preference is not set',
        'preferred language as English',
        'request interpreted as English.',
    ]
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: any(text.lower() == x.lower() for text in strings)))

def drop_cancel_or_exit(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are simply cancel or exit"""
    strings = [
        "cancel",
        "exit",
    ]
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: any(text.lower() == x.lower() for text in strings)))

def drop_asterisks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that contain >4 consecutive asterisks"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: '****' in x))

def drop_entirely_asterisks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are entirely asterisks"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: re.match(r"^\*+$", x)) == True)

def drop_empty_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that contain empty messages"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: not x or x.strip() == ""))

def drop_profanity(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that contain profanity"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: profanity.contains_profanity(x)))

def drop_testing_chats(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are testing the chatbot"""
    return drop_messages_after_condition(df, df['request_content'].apply(lambda x: ('testing' in x.lower() and 'aisera' in x.lower()) or ('testing_' in x.lower())))

def drop_duplicate_convos(df: pd.DataFrame) -> pd.DataFrame:
    """Remove conversations where the first user message is an exact duplicate of another conversation."""
    drop_conversation_uuids: List[str] = [] # list of conversation_uuids to drop
    # Collapse df to first occurrence of each conversation_uuid
    df_first = df.groupby('conversation_uuid').first().reset_index()
    first_messages = set()
    for idx, row in df_first.iterrows():
        if row['request_content'].lower() not in first_messages:
            first_messages.add(row['request_content'].lower())
        else:
            drop_conversation_uuids.append(row['conversation_uuid'])
    df = df[~df['conversation_uuid'].isin(drop_conversation_uuids)]
    return df

def drop_unknown_workflows(df: pd.DataFrame, workflow_uuids: List[str]) -> pd.DataFrame:
    """Remove chats that are associated with unknown workflows."""
    return drop_messages_after_condition(df, ~df['aisera_workflow_uuid'].isin(workflow_uuids))

ALL_FILTER_FNS = [
    # ! Note: Order matters for speed!
    drop_null_conversation_uuid,
    drop_empty_messages,
    drop_short_char_len,
    drop_cancel_or_exit,
    drop_clear_button_presses,
    drop_testing_chats,
    drop_language_preferences,
    drop_asterisks,
    drop_entirely_asterisks,
    drop_duplicate_convos,
    drop_profanity, # do last b/c very time consuming
]

def clean_html(text: str) -> str:
    """Remove HTML from the user message."""
    return re.sub(r'<[^>]*>', '', str(text))

def sample_conversation_uuids(df_messages: pd.DataFrame, df_workflows: pd.DataFrame, n_samples: int) -> List[str]:
    """Given a dataloader, sample `n_samples` unique conversation UUIDs."""
    random.seed(0)

    # Group examples by triggered workflow UUID
    df_workflow_counts = df_messages['aisera_workflow_uuid'].value_counts().reset_index()
    df_workflow_counts.columns = ['workflow_uuid', 'n_messages']
    workflow_uuids = sorted([
        x for x in df_workflow_counts[df_workflow_counts['n_messages'] > 0]['workflow_uuid'].unique().tolist()
        if (
            # remove invalid workflow UUIDs
            x not in [ None, 'nan', 'None', '']
            # only keep workflows that are in df_workflows
            and x in df_workflows['workflow_uuid'].unique().tolist()
        )
    ])
    
    # Filter dataframe to only valid workflows
    df_messages = df_messages[df_messages['aisera_workflow_uuid'].isin(workflow_uuids)]
    
    # Determine the number of chats to sample per workflow
    num_workflows: int = len(workflow_uuids)
    if num_workflows == 0:
        print("No workflows found.")
        return []
    print(f"Found {num_workflows} unique workflows.")
    per_workflow_sample: int = n_samples // num_workflows
    print(f"Attempting to sample {per_workflow_sample} chats per workflow.")
    print(f"# of messages to sample from: {len(df_messages)}")
    print(f"# of long convos to sample from: {df_messages['conversation_uuid'].apply(lambda x: len(df_messages[df_messages['conversation_uuid'] == x]) >= N_MESSAGES_LONG_CONVO_THRESHOLD).sum()}")

    # Sample chats from each workflow
    sampled_conversation_uuids: List[str] = []
    for idx, workflow_uuid in enumerate(workflow_uuids):
        df_convos = df_messages[df_messages['aisera_workflow_uuid'] == workflow_uuid]
        
        df_convo_counts = df_convos['conversation_uuid'].value_counts().reset_index()
        df_convo_counts.columns = ['conversation_uuid', 'n_messages']

        # Separate examples into long and short chats
        df_convo_counts['is_long'] = df_convo_counts['n_messages'] >= N_MESSAGES_LONG_CONVO_THRESHOLD

        # Calculate the number of long chats to sample (70%)
        num_long_chats_to_sample = int(per_workflow_sample * 0.7)
        num_short_chats_to_sample = per_workflow_sample - min(df_convo_counts['is_long'].sum(), num_long_chats_to_sample)

        # Sample long and short chats
        n_long: int = min(df_convo_counts['is_long'].sum(), num_long_chats_to_sample)
        n_short: int = min(len(df_convo_counts) - n_long, num_short_chats_to_sample)
        long_conversation_uuids: List[str] = df_convo_counts[df_convo_counts['is_long']].sample(n_long, random_state=idx)['conversation_uuid'].tolist()
        short_conversation_uuids: List[str] = df_convo_counts[~df_convo_counts['is_long']].sample(n_short, random_state=idx)['conversation_uuid'].tolist()

        print(
            f"Workflow {workflow_uuid}: "
            f"Sampling {len(long_conversation_uuids)} long chats and {len(short_conversation_uuids)} short chats."
        )

        sampled_conversation_uuids.extend(long_conversation_uuids + short_conversation_uuids)

    # If total sampled chats are less than n_samples, fill the rest
    if len(sampled_conversation_uuids) < n_samples:
        # Get all unsampled conversations
        df_convos = df_messages[~df_messages['conversation_uuid'].isin(sampled_conversation_uuids)]
        df_convo_counts = df_convos['conversation_uuid'].value_counts().reset_index()
        df_convo_counts.columns = ['conversation_uuid', 'n_messages']
        df_convo_counts['is_long'] = df_convo_counts['n_messages'] >= 4
        print(f"Sampled {len(sampled_conversation_uuids)} chats so far. Have {len(df_convo_counts)} unsampled chats remaining. Filling the rest to reach {n_samples}.")

        # Try filling with long chats
        long_conversation_uuids: List[str] = df_convo_counts[df_convo_counts['is_long']].sample(min(df_convo_counts['is_long'].sum(), n_samples - len(sampled_conversation_uuids)), random_state=0)['conversation_uuid'].tolist()
        sampled_conversation_uuids.extend(long_conversation_uuids)
        print(f"Sampled {len(long_conversation_uuids)} remaining long chats")

        # If still not enough, fill with short chats
        if len(sampled_conversation_uuids) < n_samples:
            short_conversation_uuids: List[str] = df_convo_counts[~df_convo_counts['is_long']].sample(min(len(df_convo_counts) - df_convo_counts['is_long'].sum(), n_samples - len(sampled_conversation_uuids)), random_state=0)['conversation_uuid'].tolist()
            sampled_conversation_uuids.extend(short_conversation_uuids)
            print(f"Sampled {len(short_conversation_uuids)} remaining short chats")

    return sampled_conversation_uuids

def extract_text_comprehensive(response_content: str) -> List[str]:
    """
    Extract relevant text from the `response_content` field in the `messages_info.csv` file.
    """
    try:
        parsed_data = json.loads(response_content)
        content_json = json.loads(parsed_data[0])
        extracted_texts = []

        for item in content_json:
            # Case 1: Extract text from "data" field directly
            if "data" in item:
                data = item["data"]
                if isinstance(data, str):
                    # Clean HTML tags
                    clean_text = re.sub(r'<[^>]+>', '', data)
                    extracted_texts.append(clean_text.strip())
                elif isinstance(data, dict):
                    # Case 2: Extract input table information if available
                    if "inputTable" in data:
                        for entry in data["inputTable"]:
                            if "label" in entry:
                                extracted_texts.append(entry["label"].strip())
                            if "initial_value" in entry:
                                # Check if `initial_value` is a list
                                if isinstance(entry["initial_value"], list):
                                    # Join list items into a single string or handle them individually
                                    extracted_texts.append(" ".join([str(item).strip() for item in entry["initial_value"]]))
                                elif isinstance(entry["initial_value"], str):
                                    extracted_texts.append(entry["initial_value"].strip())
                            if "help_text" in entry:
                                extracted_texts.append(entry["help_text"].strip())

                    # Case 3: Extract labels for button and dropdown items if available
                    if "items" in data:
                        labels = [label["label"].strip() for label in data["items"] if "label" in label]
                        extracted_texts.extend(labels)

                    # Case 4: Extract other relevant text fields in data
                    for key in ["submitLabel", "cancelLabel", "question"]:
                        if key in data and data[key]:
                            if isinstance(data[key], str):
                                extracted_texts.append(data[key].strip())

        return ' '.join(extracted_texts)
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        # print(f"Error processing response_content: {e}")
        return ""

def sample_domain(args, domain: str):
    """Samples chats from a single domain."""
    # Update config for this domain
    path_to_input_dir: str = os.path.join(args.path_to_dataset_dir, domain)
    if not os.path.exists(path_to_input_dir):
        print(f"Skipping company: {domain} because it doesn't exist")
        return
    
    # Update paths for this domain
    path_to_input_messages_csv: str = os.path.join(path_to_input_dir, "messages_info.csv")
    path_to_input_workflows_csv: str = os.path.join(path_to_input_dir, "workflows_info.csv")

    # Confirm that all files exist
    for path in [path_to_input_messages_csv, path_to_input_workflows_csv]:
        if not os.path.exists(path):
            raise ValueError(f"Error with {domain}: {path} doesn't exist")
    print(f"Loading dataset for {domain}...")

    # Output directory
    path_to_output_dir = os.path.join(args.path_to_output_dir, domain)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load workflows
    df_workflows = pd.read_csv(path_to_input_workflows_csv)
    df_workflows.rename(columns={
        "flow_id": "workflow_uuid",
        "fulfillment_type": "fulfillment_type",
        "flow_name": "name",
        "flow_input_variable_info": "inputs",
        "flow_output_variable_info": "output"
    }, inplace=True)
    df_workflows['domain_uuid'] = domain
    
    # Do some data cleaning of messages
    df_messages = pd.read_csv(path_to_input_messages_csv, dtype={'conversation_uuid': str, 'true_workflow_uuid': str, 'true_workflow_uuid_2': str, 'triggering_workflow_uuid' : str})
    print(f"Loaded n={len(df_messages)} raw messages from `{path_to_input_messages_csv}`")
    df_messages.rename(columns={
        'request_created_at': 'request_timestamp',
        'sender_id': 'sender_uuid',
        'conversation_id': 'conversation_uuid',
        'conversation_request_id': 'request_uuid',
        'response_content_text': 'response_content',  # Map the text content correctly
    }, inplace=True)
    df_messages = df_messages.sort_values(by=['conversation_uuid', 'id']).reset_index(drop=True)
    df_messages['request_idx'] = df_messages.groupby('conversation_uuid').cumcount() + 1
    df_messages['domain_uuid'] = domain
    # Extract text from response_content JSON blob
    df_messages['response_content'] = df_messages['response_content'].apply(extract_text_comprehensive)
    # Remove HTML
    df_messages['request_content'] = df_messages['request_content'].apply(clean_html)
    df_messages['response_content'] = df_messages['response_content'].apply(clean_html)
    # Remove rows where 'id' is not an integer
    df_messages = df_messages[df_messages['id'].apply(lambda x: str(x).isdigit())]
    
    # Dummy columns that will be filled in later during human annotation process
    df_messages['true_workflow_uuid'] = None
    df_messages['true_workflow_uuid_2'] = None
    df_messages['aisera_workflow_uuid'] = df_messages['triggering_workflow_uuid']
    df_messages['aisera_n_tokens'] = None
    df_messages['aisera_cost'] = None
    df_messages['aisera_latency'] = None

    df_messages = df_messages.astype(dtype={
        'id' : int,
        'depth_of_conversation' : int,
        'request_idx' : int,
        'domain_uuid': str,
        'sender_uuid': str,
        'request_uuid': str,
        'conversation_uuid': str,
        'true_workflow_uuid': str,
        'true_workflow_uuid_2': str,
        'aisera_workflow_uuid': str,
        'triggering_workflow_uuid' : str,
        'aisera_n_tokens': str,
        'aisera_cost': str,
        'aisera_latency': str,
        'request_content' : str,
        'request_timestamp': str,
        'response_content': str,
    })

    # Remove rows with invalid Unicode characters
    def fix_text(text):
        if isinstance(text, str):
            # Re-encode using surrogatepass then decode with error replacement
            return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        return text
    for col in df_messages.select_dtypes(include=["object"]).columns:
        df_messages[col] = df_messages[col].apply(fix_text)
    
    # Filter out low quality conversations
    drop_stats = collections.defaultdict(dict) # [key] = domain, [value] = dict of drop stats
    start_n_rows = df_messages.shape[0]
    print("-" * 100)
    print(f"Filtering {domain}")
    print(f"=> Start | n_rows={start_n_rows}")
    print(f"  Filter{'':<47} | dropped_rows | time (s)")
    print("  " + "-" * 70)
    for filter_fn in ALL_FILTER_FNS:
        old_n_rows = df_messages.shape[0]
        start_time = time.time()
        df_messages = filter_fn(df_messages)
        drop_stats[domain][filter_fn.__name__] = old_n_rows - df_messages.shape[0]
        print(f"   {filter_fn.__name__:<50} | {old_n_rows - df_messages.shape[0]:>5} | {time.time() - start_time:.2f}")
    print("  " + "-" * 70)
    print(f"   {'Total':<50} | {start_n_rows - df_messages.shape[0]:>5}")
    print(f"    Final stats:")
    print(f"      - start_rows={start_n_rows}")
    print(f"      - end_rows={df_messages.shape[0]}")
    print(f"      - dropped_rows={start_n_rows - df_messages.shape[0]} ({(start_n_rows - df_messages.shape[0]) / start_n_rows * 100:.2f}%)")

    # Sample conversation UUIDs
    print(f"Sampling {args.n_samples} chats from domain {domain}...")
    conversation_uuids: List[str] = sample_conversation_uuids(df_messages, df_workflows, n_samples=args.n_samples)
    df_messages = df_messages[df_messages["conversation_uuid"].isin(conversation_uuids)].copy()
    print(f"After sampling, n={len(df_messages)} messages remain | sampled {len(conversation_uuids)} conversation UUIDs")

    # Deidentify messages
    print(f"De-identifying messages...")
    start_time = time.time()
    df_messages['request_content'] = df_messages['request_content'].apply(lambda x: deidentify_text(x, domain))
    df_messages['response_content'] = df_messages['response_content'].apply(lambda x: deidentify_text(x, domain))
    print(f"Done de-identifying messages. Time taken: {time.time() - start_time:.2f} seconds")
    df_messages = df_messages[DF_MESSAGE_COLUMNS]
    ## Drop conversations where >=50% of messages contain 'REDACTED'
    df_messages = drop_redacted_conversations(df_messages)
    df_messages = drop_duplicate_convos(df_messages) # re-run after deidentifying
    
    # Save messages
    df_messages.to_csv(os.path.join(path_to_output_dir, "messages.csv"), index=False)
    print(f"Saved messages to `{os.path.join(path_to_output_dir, 'messages.csv')}`")

    # Save df_domains.csv
    df_domains = pd.DataFrame({
        'domain_uuid': [domain],
        'domain_name': [domain],
    })
    df_domains.to_csv(os.path.join(path_to_output_dir, "domains.csv"), index=False)
    print(f"Saved domains to `{os.path.join(path_to_output_dir, 'domains.csv')}`")

    # Deidentify workflows
    print(f"De-identifying workflows...")
    df_workflows['description'] = df_workflows['description'].apply(lambda x: deidentify_text(x, domain))
    df_workflows['name'] = df_workflows['name'].apply(lambda x: deidentify_text(x, domain))
    
    # Save df_workflows.csv
    df_workflows.to_csv(os.path.join(path_to_output_dir, "workflows.csv"), index=False)
    print(f"Saved workflows to `{os.path.join(path_to_output_dir, 'workflows.csv')}`")

def print_stats(args, domains: List[str]):
    """Prints statistics on the sampled dataset."""
    print("\nOverall Statistics\n")
    rows = []
    for domain in domains:
        df_messages = pd.read_csv(os.path.join(args.path_to_output_dir, domain, 'messages.csv'))
        df_workflows = pd.read_csv(os.path.join(args.path_to_output_dir, domain, 'workflows.csv'))
        rows.append([
            domain, 
            df_messages.shape[0], 
            df_messages['conversation_uuid'].nunique(), 
            df_messages['conversation_uuid'].apply(lambda x: len(df_messages[df_messages['conversation_uuid'] == x]) >= N_MESSAGES_LONG_CONVO_THRESHOLD).sum(),
            df_workflows.shape[0]
        ])
    SEP = '=' * 12
    rows.append([SEP, SEP, SEP, SEP, SEP])
    rows.append([ 'Total', 
                 sum([row[1] for row in rows if row[1] != SEP]), 
                 sum([row[2] for row in rows if row[2] != SEP]), 
                 sum([row[3] for row in rows if row[3] != SEP]),
                 sum([row[4] for row in rows if row[4] != SEP])])
    print(tabulate(rows, 
                   headers=['Domain', '# Messages', '# Convos', '# Long Convos', '# Workflows'], 
                   tablefmt='orgtbl'))
    print("Done!")

def process_domain(domain_args):
    """Wrapper function to unpack arguments for sample_domain"""
    args, domain = domain_args
    return sample_domain(args, domain)

def main():
    args = parse_args()
    
    # Process domains
    DOMAIN_LIST: List[str] = [ x for x in os.listdir(args.path_to_dataset_dir) if os.path.isdir(os.path.join(args.path_to_dataset_dir, x)) ]
    print(f"Found {len(DOMAIN_LIST)} domains to sample from: {DOMAIN_LIST}")

    for domain in tqdm(DOMAIN_LIST, desc=f"Sampling {args.n_samples} chats from each domain", total=len(DOMAIN_LIST)):
        sample_domain(args, domain)
    
    # Print statistics on each domain
    print_stats(args, DOMAIN_LIST)

if __name__ == "__main__":
    main()
