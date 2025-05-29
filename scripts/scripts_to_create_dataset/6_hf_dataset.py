"""
Usage:
    python 6_hf_dataset.py --path_to_dataset_dir <path_to_dataset_dir> --hf_version <hf_version>

Example:
    python 6_hf_dataset.py --path_to_dataset_dir data/3_clean_v1 --path_to_output_dir data/4_hf_dataset_v1 --hf_version v1

Purpose:
    Create a dataset for the Hugging Face Hub.
"""
import os
import shutil
import pandas as pd
import argparse
from huggingface_hub import upload_folder
import datetime
from classicbench.utils import get_rel_path

def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--path_to_dataset_dir", type=str, default=get_rel_path("data/3_clean_v1"), help="Path to the dataset directory. This should contain multiple .xlsx files, each .xlsx file with the same structure but from a different annotator.")
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("data/4_hf_dataset_v1"), help="Path to the output directory.")
    parser.add_argument("--hf_version", type=str, default="v1", help="Version of the dataset on the Hugging Face Hub.")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = f"Miking98/classic_benchmark-{args.hf_version}"
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Copy datasets to output directory
    for file in os.listdir(args.path_to_dataset_dir):
        if file.endswith(".csv"):
            shutil.copy(os.path.join(args.path_to_dataset_dir, file), os.path.join(args.path_to_output_dir, file))
    
    # Load datasets
    df_messages = pd.read_csv(os.path.join(args.path_to_output_dir, f"messages.csv"))
    df_workflows = pd.read_csv(os.path.join(args.path_to_output_dir, f"workflows.csv"))
    df_jailbreak_prompts = pd.read_csv(os.path.join(args.path_to_output_dir, f"jailbreak_prompts.csv"))
    df_domains = pd.read_csv(os.path.join(args.path_to_output_dir, f"domains.csv"))
    
    # Drop Aisera stats
    df_messages = df_messages.drop(columns=['aisera_workflow_uuid', 'aisera_n_tokens', 'aisera_cost', 'aisera_latency', ])
    df_messages.to_csv(os.path.join(args.path_to_output_dir, f"messages.csv"), index=False)
    
    # Statistics
    ## Per domain
    stats_per_domain = {}
    for domain_uuid in df_domains["domain_uuid"].unique():
        n_messages = df_messages[df_messages["domain_uuid"] == domain_uuid].shape[0]
        n_conversations = df_messages[df_messages["domain_uuid"] == domain_uuid]["conversation_uuid"].nunique()
        n_workflows = df_workflows[df_workflows["domain_uuid"] == domain_uuid]["workflow_uuid"].nunique()
        stats_per_domain[domain_uuid] = {
            'domain' : domain_uuid,
            "n_messages": n_messages,
            "n_conversations": n_conversations,
            "n_workflows": n_workflows,
        }
    stats_per_domain['Total'] = {
        'domain' : 'Total',
        "n_messages": df_messages.shape[0],
        "n_conversations": df_messages["conversation_uuid"].nunique(),
        "n_workflows": df_workflows["workflow_uuid"].nunique(),
    }
    stats_per_domain = pd.DataFrame(stats_per_domain).T
    stats_per_domain.rename(columns={'domain' : 'Domain', 'n_messages' : 'Messages', 'n_conversations' : 'Conversations', 'n_workflows' : 'Workflows'}, inplace=True)
    ## Totals
    n_messages = df_messages.shape[0]
    n_conversations = df_messages["conversation_uuid"].nunique()
    n_workflows = df_workflows["workflow_uuid"].nunique()
    n_jailbreak_prompts = df_jailbreak_prompts.shape[0]
    n_domains = df_domains["domain_uuid"].nunique()
    
    # README file
    readme_content = f"""---
license: apache-2.0
task_categories:
- text-classification
language:
- en
tags:
- agent
- enterprise
configs:
- config_name: messages
  data_files: 
  - split: test
    path: "messages.csv"
- config_name: workflows
  data_files: 
  - split: test
    path: "workflows.csv"
- config_name: jailbreak_prompts
  data_files: 
  - split: test
    path: "jailbreak_prompts.csv"
- config_name: domains
  data_files: 
  - split: test
    path: "domains.csv"
---
  
# CLASSic Benchmark ({args.hf_version})

Version **{args.hf_version}** of the CLASSic Benchmark. Uploaded on {datetime.datetime.now().strftime("%B %d, %Y")}. 

Please see [Github](https://github.com/Miking98/classic_benchmark) for more information and model leaderboards.

*Note: This is a filtered subset of the dataset originally published in the [CLASSIC Benchmark ICLR 2025 Workshop paper](https://openreview.net/forum?id=RQjUpeINII) which was cleared for public release.*

## Usage

```python
from datasets import load_dataset

# Load dataset subsets
ds_messages = load_dataset('{dataset_name}', 'messages')
ds_workflows = load_dataset('{dataset_name}', 'workflows')
ds_domains = load_dataset('{dataset_name}', 'domains')
ds_jailbreak_prompts = load_dataset('{dataset_name}', 'jailbreak_prompts')
```

## Statistics

### Total Counts
- \# of messages: {n_messages}
- \# of conversations: {n_conversations}
- \# of workflows: {n_workflows}
- \# of jailbreak prompts: {n_jailbreak_prompts}
- \# of domains: {n_domains}

### Per Domain Counts
{stats_per_domain.to_markdown(index=False)}

## Description

This dataset contains four subsets:
- **messages**: Messages from real-world user-chatbot conversations
- **workflows**: Workflow names and descriptions for each domain (i.e. the classes being predicted in this task)
- **domains**: Domain names and descriptions
- **jailbreak_prompts**: Jailbreak prompts used for *security* evaluation
"""

    # Create README file
    with open(os.path.join(args.path_to_output_dir, "README.md"), "w") as f:
        f.write(readme_content)
        
    # Upload directory to the repository
    upload_folder(
        folder_path=args.path_to_output_dir,
        repo_id=dataset_name,
        repo_type="dataset",
    )
    

if __name__ == "__main__":
    main()
