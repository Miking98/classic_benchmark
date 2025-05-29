"""Usage:
    python3 2_convert_sampled_to_annotations.py

Purpose:
    Convert sampled chats to Excel sheet for AMT annotations.
"""
import argparse
import os
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import base64
from classicbench.utils import get_rel_path
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Sample and Evaluate Chat Data")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/1_sampled"), help="Path to the raw dataset dump from Aisera.")
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("data/2_annotations"), help="Path to where sampled chats will be saved.")
    parser.add_argument("--is_force_refresh", action="store_true", help="Force refresh the cache.")
    parser.add_argument("--n_samples", type=int, default=500, help="# of samples to take from the dataset.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Excel columns
    CHATS_EXCEL_COLUMNS: List[str] = [
        "Is gibberish?",
        "Is asking for live human agent / call representative?",
        "Is profanity?",
        "Is PII? (e.g phone number, email, etc.)",
        "Is bot response contain list of options?",
        "Workflow ID (1)",
        "Workflow ID (2)",
    ]
    WORKFLOWS_COLUMN_MAPPING: Dict[str, str] = {
        "workflow_uuid" : "ID",
        "name" : "Name",
        "description" : "Description",
    }
    sheets: Dict[str, pd.DataFrame] = {}
    
    # Process domains
    DOMAIN_LIST: List[str] = [
        'medical', 
        'hr', 
        'it',
        'finance',
        'biotech',
        'bank',
        'edtech',
    ]
    for domain in tqdm(DOMAIN_LIST, desc=f"Converting sampled chats to Excel sheet for AMT annotations", total=len(DOMAIN_LIST)):
        path_to_message_csv: str = os.path.join(args.path_to_dataset_dir, domain, f"messages.csv")
        path_to_workflow_csv: str = os.path.join(args.path_to_dataset_dir, domain, f"workflows.csv")
        
        df_messages = pd.read_csv(path_to_message_csv)
        df_workflows = pd.read_csv(path_to_workflow_csv)
        
        # Messages
        df_messages.rename(columns={
            "conversation_uuid": "Conversation ID",
            "request_content": "User Message",
            "response_content": "Bot Response",
            "request_uuid" : "request_uuid",
            "request_idx" : "request_idx",
            "triggering_workflow_uuid": "triggering",
        }, inplace=True)
        for col in CHATS_EXCEL_COLUMNS:
            df_messages[col] = ""
        df_messages = df_messages.fillna("")
        df_messages = df_messages[[ *CHATS_EXCEL_COLUMNS, "Conversation ID", "User Message", "Bot Response", "request_uuid", "request_idx", "triggering"]]
        ## Hide bot triggered workflow from annotators
        df_messages["triggering"] = df_messages["triggering"].apply(lambda x: base64.b64encode(x.encode('utf-8')).decode('utf-8'))

        # Workflows
        df_workflows.rename(columns=WORKFLOWS_COLUMN_MAPPING, inplace=True)
        df_workflows = df_workflows[WORKFLOWS_COLUMN_MAPPING.values()]
        df_workflows = df_workflows.fillna("")
        df_workflows["Description"] = df_workflows["Description"].apply(lambda x: x[:500]) # limit to 500 characters
        ## Add workflow at top row for "None"
        df_workflows = pd.concat([pd.DataFrame({
            "ID": ["None"],
            "Name": ["None"],
            "Description": ["No other workflow fits"],
        }), df_workflows])
        
        # Save sheets
        sheets[domain] = df_messages.copy()
        sheets[f"{domain}_workflows"] = df_workflows.copy()
    
    # Save sheets as Excel file
    path_to_output_excel: str = os.path.join(args.path_to_output_dir, "annotations.xlsx")
    with pd.ExcelWriter(path_to_output_excel, engine='xlsxwriter') as writer:
        # First, create title sheet
        df_title = pd.DataFrame({
            "Instructions": [
                "Thank you for taking the time to help with dataset annotation for the AI chat bot benchmark.",
                "",
                "Instructions:",
                "",
                "1. Only fill out these columns:",
                "    - Is gibberish?",
                "    - Is asking for live human agent / call representative?", 
                "    - Is profanity?",
                "    - Is PII? (e.g phone number, email, etc.)",
                "    - Is bot response contain list of options?",
                "    - Workflow ID (1)",
                "    - Workflow ID (2)",
                "",
                "2. A conversation can span 1+ rows.",
                "   Each row contains a single user message and the chat bot's response.",
                "   Messages belonging to the same conversation will have the same \"Conversation ID\".",
                "   Messages in the same conversation will be listed in chronological order from top to bottom.",
                "",
                "3. For instructions, please see:",
                "   https://docs.google.com/document/d/1f5Unv4g9iYSSsuS4s99Mu9_VrgsZkWTnNsZBawWc-tg/edit?tab=t.0"
            ]
        })
        df_title.to_excel(writer, sheet_name="Instructions", index=False)
        # Second, create sheet for each domain
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                # Get the maximum length of the column name and its contents
                max_length = max(
                    df[col].astype(str).apply(len).max(),  # max length of column contents
                    len(str(col))  # length of column name
                )
                # Add a little extra space
                adjusted_width: int = min(max_length + 2, 50)
                # Set the column width
                worksheet.set_column(idx, idx, adjusted_width)
    print(f"Saved annotations to `{path_to_output_excel}`")

if __name__ == "__main__":
    main()
