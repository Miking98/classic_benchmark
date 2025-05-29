import re
from typing import List
import pandas as pd
import os

def process_csv_files(path_to_messages_csv):
    """
    Process CSV files to clean data and create domain-specific outputs.
    
    Args:
        annotated_path: Path to the annotated.csv file
        messages_path: Path to the messages.csv file
    """
    
    df = pd.read_csv(path_to_messages_csv)
    
    path_to_messages_csv = path_to_messages_csv.replace(".csv", "_cleaned.csv")

    # Clean the data
    print("Original shape:", df.shape)
    df = drop_duplicate_convos(df)
    print("After dropping duplicates:", df.shape)
    df = clean_html(df)
    print("After cleaning HTML:", df.shape)
    df = drop_clear_button_presses(df)
    print("After dropping clear button presses:", df.shape)
    df = drop_redacted_data(df)
    print("After dropping redacted data:", df.shape)
    df = drop_testing_chats(df)
    print("After dropping testing chats:", df.shape)
    df = keep_necessary_columns(df)
    print("After keeping necessary columns:", df.shape)
    
    # Save final result back to original messages.csv
    df.to_csv(path_to_messages_csv, index=False)
    
    print("Final shape:", df.shape)
    
    # Create domain-specific files
    create_domain_files(df, os.path.dirname(path_to_messages_csv))

def drop_duplicate_convos(df: pd.DataFrame) -> pd.DataFrame:
    """Remove conversations where the first user message is an exact duplicate of another conversation."""
    first_messages = set()
    drop_conversation_uuids: List[str] = [] # list of conversation_uuids to drop
    # Collapse df to first occurrence of each conversation_uuid
    df_first = df.groupby('conversation_uuid').first().reset_index()
    for idx, row in df_first.iterrows():
        if row['request_content'].lower() not in first_messages:
            first_messages.add(row['request_content'].lower())
        else:
            drop_conversation_uuids.append(row['conversation_uuid'])
    df = df[~df['conversation_uuid'].isin(drop_conversation_uuids)]
    return df

def drop_testing_chats(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chats that are simply testing the chatbot, e.g. 'testing by aisera' or 'Testing_cancel_ticket'"""
    df['is_invalid'] = df['request_content'].apply(lambda x: ('testing' in x.lower() and 'aisera' in x.lower()) or ('testing_' in x.lower()))
    df = df[~df['is_invalid']]
    df = df.drop(columns=['is_invalid'])
    return df

def clean_html(df: pd.DataFrame) -> pd.DataFrame:
    """Remove HTML from the user message."""
    df['request_content'] = df['request_content'].apply(lambda x: re.sub(r'<[^>]*>', '', str(x)))
    return df

def drop_redacted_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove redacted data."""
    df['is_redacted'] = df['request_content'].apply(lambda x: x == '[REDACTED]')
    df = df[~df['is_redacted']]
    df = df.drop(columns=['is_redacted'])
    return df

def drop_clear_button_presses(df: pd.DataFrame) -> pd.DataFrame:
    """Remove clear button presses -- e.g. Account Management, Direct Deposit, Check Direct Deposit Status."""
    button_press_texts = [
        "Account Management",
        "Direct Deposit",
        "Check Direct Deposit Status",
    ]
    df['is_invalid'] = df['request_content'].apply(lambda x: any(text.lower() == x.lower() for text in button_press_texts))
    df = df[~df['is_invalid']]
    df = df.drop(columns=['is_invalid'])
    return df

def keep_necessary_columns(df):
    """Keep only the necessary columns in the specified order"""
    columns_to_keep = [
        'conversation_uuid',
        'request_content',
        'response_content',
        'true_workflow_uuid',
        'true_workflow_uuid_2',
        'aisera_workflow_uuid',
        'aisera_n_tokens',
        'aisera_cost',
        'aisera_latency',
        'request_idx',
        'domain_uuid'
    ]
    
    return df[columns_to_keep]

def create_domain_files(df, base_dir):
    """Create domain-specific CSV files"""
    unique_domains = df['domain_uuid'].unique()
    
    print("\nCreating domain-specific files:")
    for domain in unique_domains:
        # Create domain directory
        domain_dir = os.path.join(base_dir, str(domain))
        os.makedirs(domain_dir, exist_ok=True)
        
        # Filter data for this domain
        domain_df = df[df['domain_uuid'] == domain]
        
        # Save to messages.csv in the domain directory
        output_path = os.path.join(domain_dir, 'messages.csv')
        domain_df.to_csv(output_path, index=False)
        
        print(f"- Domain {domain}: {len(domain_df)} rows saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process and clean conversation data')
    parser.add_argument('messages', help='Path to messages.csv file')
    args = parser.parse_args()
    
    process_csv_files(args.messages) 