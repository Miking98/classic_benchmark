"""
Usage:
    python3 scripts/figures/print_dataset_stats.py

Purpose:
    Print stats about dataset.
"""
import pandas as pd
import argparse
import os

from classicbench.utils import get_rel_path

def parse_args():
    parser = argparse.ArgumentParser(description="Get stats about dataset")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/3_clean_iclr_v0"), help="Path to the dataset directory. This should contain a `messages.csv` file, `domains.csv` file, and `workflows.csv` file.")
    return parser.parse_args()

def main():
    args = parse_args()

    df_messages = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"messages.csv"))
    df_domains = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"domains.csv"))
    df_workflows = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"workflows.csv"))
    
    ############################################################
    # Messages
    ############################################################
    print("-" * 20 + " Messages " + "-" * 20)
    df_messages['content_length'] = df_messages['request_content'].str.len()
    df_messages['label_count'] = df_messages['true_workflow_uuid'].notna().astype(int) + df_messages['true_workflow_uuid_2'].notna().astype(int)
    print(f"\nNumber of messages: {len(df_messages)}")
    print(f"\nNumber of conversations: {len(df_messages['conversation_uuid'].unique())}")
    print(f"\nNumber of messages per conversation:\n{df_messages.groupby('conversation_uuid').size().describe()}")
    print(f"\nNumber of messages per domain:\n{df_messages.value_counts('domain_uuid')}")
    print(f"\nNumber of conversations per domain:\n{df_messages.groupby('domain_uuid').agg({ 'conversation_uuid' : lambda x: len(set(x)) })}")
    print(f"\nNumber of single-message conversations: {(df_messages.value_counts('conversation_uuid').reset_index()['count'] == 1).sum()}")
    print(f"\nNumber of multi-message conversations: {(df_messages.value_counts('conversation_uuid').reset_index()['count'] > 1).sum()}")
    print(f"\nCharacters per message per domain:\n{df_messages.groupby(['domain_uuid']).agg({ 'content_length' : 'mean' })}")
    print(f"\nLabel count per domain:\n{df_messages.groupby(['domain_uuid', 'label_count']).size().reset_index()}")
    print(f"\nCharacters per message:\n{df_messages['content_length'].mean()}")
    print(f"\nLabel counts:\n{df_messages['label_count'].value_counts()}")
    print(f"\nNumber of messages per intent:\n{df_messages.groupby(['true_workflow_uuid']).size().reset_index().describe()}")
    print(f"\nNumber of conversations per intent:\n{df_messages.groupby(['true_workflow_uuid']).agg({ 'conversation_uuid' : lambda x: len(set(x)) }).describe()}")
    print(f"\nNumber of intents per domain:\n{df_messages.groupby(['domain_uuid']).agg({ 'true_workflow_uuid' : 'nunique' }).describe()}")

    ############################################################
    # Domains
    ############################################################
    print()
    print("-" * 20 + " Domains " + "-" * 20)
    print(f"Number of domains: {len(df_domains)}")
    df_intents = df_messages.groupby(['true_workflow_uuid']).agg({ 'domain_uuid' : 'first', }).reset_index()
    print("Intents per domain:\n", df_intents.groupby('domain_uuid').agg({ 'true_workflow_uuid' : 'count' }))

    ############################################################
    # Workflows
    ############################################################
    print()
    print("-" * 20 + " Workflows " + "-" * 20)
    print(f"\nNumber of workflows: {len(df_workflows)}")
    print(f"\nNumber of workflows per domain:\n{df_workflows.value_counts('domain_uuid')}")
    
    ############################################################
    # Labels
    ############################################################
    print()
    print("-" * 20 + " Labels " + "-" * 20)
    print(f"\nNumber of messages WITH true workflow: {len(df_messages[df_messages['true_workflow_uuid'].notna()])}")
    print(f"\nNumber of messages WITHOUT true workflow: {len(df_messages[df_messages['true_workflow_uuid'].isna()])}")
    print(f"\nNumber of messages WITH aisera workflow: {len(df_messages[df_messages['aisera_workflow_uuid'].notna()])}")

if __name__ == "__main__":
    main()
