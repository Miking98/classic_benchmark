
"""
Usage:
    python3 4_convert_clean_to_iclr_workshop_sample.py

Purpose:
    Create a small subsample of the cleaned dataset for the ICLR Workshop submission.
"""
import argparse
import os
import pandas as pd
from classicbench.utils import get_rel_path

def parse_args():
    parser = argparse.ArgumentParser(description="Create ICLR Workshop Dataset Sample")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/3_clean"), help="Path to the dataset directory. This should contain multiple .xlsx files, each .xlsx file with the same structure but from a different annotator.")
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("data/iclr_workshop_sample"), help="Path to the output directory. This will contain the filtered .csv files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df_messages = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"messages.csv"))
    df_workflows = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"workflows.csv"))
    df_jailbreak_prompts = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"jailbreak_prompts.csv"))
    df_domains = pd.read_csv(os.path.join(args.path_to_dataset_dir, f"domains.csv"))
    
    # Perform stratified sampling of 10 rows per domain_uuid
    df_messages = df_messages.groupby('domain_uuid', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 10), random_state=42)
    ).reset_index(drop=True)
    
    # Only keep workflows that are in df_messages
    df_workflows = df_workflows[df_workflows['workflow_uuid'].isin(df_messages['true_workflow_uuid'].unique())]

    os.makedirs(args.path_to_output_dir, exist_ok=True)
    df_messages.to_csv(os.path.join(args.path_to_output_dir, f"messages.csv"), index=False)
    df_workflows.to_csv(os.path.join(args.path_to_output_dir, f"workflows.csv"), index=False)
    df_jailbreak_prompts.to_csv(os.path.join(args.path_to_output_dir, f"jailbreak_prompts.csv"), index=False)
    df_domains.to_csv(os.path.join(args.path_to_output_dir, f"domains.csv"), index=False)
    print(f"Created sample of {len(df_messages)} messages")
