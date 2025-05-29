"""
Usage:
    python3 run.py --is_force_refresh --data configs/data/real.yaml --agent configs/agent/cot_gpt4.yaml --eval configs/eval/default.yaml
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import pandas as pd
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from classicbench.agents.base import Agent
from classicbench.data.base import Dataloader, Dataset, PredictionWithMetadata
from classicbench.eval.metrics import calc_accuracy, calc_cost, calc_latency, calc_security, calc_stability
from classicbench.loaders.data import load_dataloader, load_dataset
from classicbench.loaders.agent import load_agent
from classicbench.utils import flatten_dict, get_rel_path, override_config_with_cli_args, load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Eval a model")
    # Dataset
    parser.add_argument( "--data", type=str, default=get_rel_path("configs/data/real.yaml"), help="Name of config YAML file in configs/data/ to load for this agent." )
    # Agent
    parser.add_argument( "--agent", type=str, default=get_rel_path("configs/agent/default.yaml"), help="Name of config YAML file in configs/evals/ to load for this agent." )
    # Evaluation
    parser.add_argument( "--eval", type=str, default=get_rel_path("configs/eval/default.yaml"), help="Name of config YAML file in configs/agents/ to load for this agent." )
    # Outputs
    parser.add_argument( "--run_uuid", type=str, default=str(int(time.time() * 1000)), help="Run UUID." )
    parser.add_argument( "--is_force_refresh", action="store_true", help="Force refresh the cache." )
    parser.add_argument( "--is_debug", action="store_true", help="Debug mode." )
    parser.add_argument( "--n_workers", type=int, default=10, help="Number of threads to use for parallel evaluation." )
    parser.add_argument( "--path_to_output_dir", type=str, default=get_rel_path("ignore/outputs/"), help="Path to where outputs / metrics will be saved." )
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def _run_helper(args) -> List[Dict]:
    """For parallel evaluation of a single example."""
    example_idx, example, config, agent, dataset = args
    domain_uuid: str = example.x.domain_uuid
    conversation_uuid: str = example.x.conversation_uuid
    message_uuid: str = example.x.messages[-1].message_uuid

    # Sanity checks. Note that sometimes the cached dataloader gets out of sync with the dataset, which this should catch.
    assert domain_uuid in dataset.df_domains['domain_uuid'].values, f"Domain {domain_uuid} not found in dataset."
    assert conversation_uuid in dataset.df_messages['conversation_uuid'].values, f"Conversation {conversation_uuid} not found in dataset."
    assert message_uuid in dataset.df_messages['message_uuid'].values, f"Message {message_uuid} not found in dataset."
    
    # Run evals
    results = []
    for trial_idx in range(config.eval.n_trials):
        pred: PredictionWithMetadata = agent.predict(example)
        predicted_workflow_uuid: str = pred.prediction.workflow.workflow_uuid if (pred.prediction and pred.prediction.workflow is not None) else None
        results.append({
            'example_idx' : example_idx,
            'trial_idx' : trial_idx,
            # prediction
            'pred_workflow_uuid' : predicted_workflow_uuid, # ACCURACY
            # label(s)
            'y_workflow_uuids' : [ x.workflow_uuid for x in example.true_y ], # List of workflows
            # metrics
            'time_to_pred' : pred.time_to_pred, # LATENCY
            'tokens_input' : pred.tokens_input, # COST
            'tokens_output' : pred.tokens_output, # COST
            'cost' : pred.cost, # COST
            # chat metadata
            'conversation_uuid' : conversation_uuid,
            'domain_uuid' : domain_uuid,
            'most_recent_message_uuid' : message_uuid,
            'most_recent_message_content' : example.x.messages[-1].content,
            'most_recent_message_idx' : example.x.messages[-1].message_idx,
            # agent metadata
            'config_agent' : str(config.agent),
            'config_eval' : str(config.eval),
        })
    return results

def calculate_metrics(df: pd.DataFrame, 
                      config: DictConfig, 
                      dataset: Dataset, 
                      path_to_run_output_dir: str, 
                      file_prefix: str, 
                      n_workers: int, 
                      is_force_refresh: bool = False, 
                      is_debug: bool = False, 
                      is_print: bool = False) -> None:
    """Calculate metrics."""
    # Paths
    path_to_metrics_json = os.path.join(path_to_run_output_dir, f'{file_prefix}_metrics.json')
    path_to_metrics_csv = os.path.join(path_to_run_output_dir, f'{file_prefix}_metrics.csv')
    
    # Calculate metrics
    limit_to_metrics = [ 'cost', 'latency', 'accuracy', 'stability', 'security' ] if (not hasattr(config.eval, 'limit_to_metrics') or config.eval.limit_to_metrics is None) else config.eval.limit_to_metrics
    metrics = {
        'n_rows' : df.shape[0],
        'cost' : calc_cost(df) if 'cost' in limit_to_metrics else None,
        'latency' : calc_latency(df) if 'latency' in limit_to_metrics else None,
        'accuracy' : calc_accuracy(df) if 'accuracy' in limit_to_metrics else None,
        'stability' : calc_stability(df) if 'stability' in limit_to_metrics else None,
        'security' : calc_security(dataset.df_jailbreakprompts, config, n_workers, is_debug) if 'security' in limit_to_metrics else None,
    }
    # If existing JSON file exists + is_force_refresh is False, merge existing metrics with new metrics
    if os.path.exists(path_to_metrics_json) and not is_force_refresh:
        existing_metrics = json.load(open(path_to_metrics_json))
        metrics = { key: val for key, val in metrics.items() if val is not None }
        metrics = { **existing_metrics, **metrics } # merge existing metrics with new metrics
    json.dump(metrics, open(path_to_metrics_json, 'w'), indent=2)
    print(f"Saved metrics JSON to `{path_to_metrics_json}`")

    ## Flatten JSON and save as CSV
    df_metrics = pd.DataFrame(flatten_dict(metrics) | { 'config' : str(config) }, index=[0])
    df_metrics.to_csv(path_to_metrics_csv, index=False)
    print(f"Saved metrics CSV to `{path_to_metrics_csv}`")

    # Print to console
    if is_print:
        print("==== Metrics Start ====")
        print(json.dumps(metrics, indent=2))
        print("==== Metrics End ====")

def main():
    args, unknown_args = parse_args()
    
    # Remap config paths if filenames are provided
    if not os.path.exists(args.data):
        args.data = os.path.join('configs/data', args.data + '.yaml')
    if not os.path.exists(args.agent):
        args.agent = os.path.join('configs/agent', args.agent + '.yaml')
    if not os.path.exists(args.eval):
        args.eval = os.path.join('configs/eval', args.eval + '.yaml')
    
    # Logging
    path_to_run_output_dir: str = os.path.join(args.path_to_output_dir, args.run_uuid)
    os.makedirs(path_to_run_output_dir, exist_ok=True)

    # Save args
    print('==> Args:', args, unknown_args)
    json.dump({'args' : vars(args), 'unknown_args' : unknown_args }, open(os.path.join(path_to_run_output_dir, 'args.json'), 'w'), indent=2)
    
    ###################################################
    # Load configs
    config: DictConfig = load_config(args)
    config.run_uuid = args.run_uuid
    
    # Override config with CLI args
    config = override_config_with_cli_args(config, unknown_args)
    print('==> Config:', config)
    json.dump(OmegaConf.to_container(config, resolve=True), open(os.path.join(path_to_run_output_dir, 'config.json'), 'w'), indent=2)

    ###################################################
    # Load dataset
    dataset: Dataset = load_dataset(config, is_force_refresh=args.is_force_refresh)
    print(f"Loaded dataset with {len(dataset.df_domains)} domains, {len(dataset.df_workflows)} workflows, {len(dataset.df_messages) // 2} messages, {len(dataset.df_messages['conversation_uuid'].unique())} conversations")
    
    # Subset dataset
    dataset = Dataset.filter(dataset, config)
    print(f"Filtered dataset to {len(dataset.df_domains)} domains, {len(dataset.df_workflows)} workflows, {len(dataset.df_messages) // 2} messages, {len(dataset.df_messages['conversation_uuid'].unique())} conversations")
    
    # Load dataloader
    dataloader: Dataloader = load_dataloader(config, dataset, is_use_cache=not args.is_force_refresh)
    print(f"Loaded dataloader with {len(dataloader)} examples.")
    
    # Sanity check
    print(dataloader[0])

    ###################################################
    # Load agent
    agent: Agent = load_agent(config)
    print(f"Loaded agent `{agent.__class__.__name__}`")

    ###################################################
    # Run evals
    path_to_results_csv: str = os.path.join(path_to_run_output_dir, 'results.csv')
    if os.path.exists(path_to_results_csv) and not args.is_force_refresh:
        print(f"Skipping running evals because `results.csv` exists and --is_force_refresh is False. Loading results from `{path_to_results_csv}`")
        df_results = pd.read_csv(path_to_results_csv)
    else:
        df_results = []
        if args.n_workers > 1:
            with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                futures = [ 
                    executor.submit(_run_helper, (example_idx, example, config, agent, dataset)) 
                    for example_idx, example in enumerate(dataloader) 
                    if (not args.is_debug or example_idx < 10) # limit to 10 examples for debugging
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Running evals..."):
                    df_results.extend(future.result())
        else:
            for example_idx, example in tqdm(enumerate(dataloader), total=len(dataloader), desc="Running evals..."):
                if (not args.is_debug or example_idx < 10): # limit to 10 examples for debugging
                    df_results.extend(_run_helper((example_idx, example, config, agent, dataset)))

        # Save results
        json.dump(df_results, open(os.path.join(path_to_run_output_dir, 'results.json'), 'w'), indent=2)
        df_results = pd.DataFrame(df_results)
        df_results = df_results.astype({ 
            'conversation_uuid' : 'str', 
            'domain_uuid' : 'str', 
            'most_recent_message_uuid' : 'str', 
            'most_recent_message_content' : 'str', 
            'config_agent' : 'str',
        })
        df_results.to_csv(path_to_results_csv, index=False)
        print(f"Saved results to `{path_to_results_csv}`")
        print(f"Saved results to `{os.path.join(path_to_run_output_dir, 'results.json')}`")

    ###################################################
    # Calculate metrics
    calculate_metrics(df_results, config, dataset, path_to_run_output_dir, 'overall', args.n_workers, is_force_refresh=args.is_force_refresh, is_debug=args.is_debug, is_print=False)

    ###################################################
    # Calculate metrics per domain
    for domain_uuid in dataset.df_domains['domain_uuid'].unique():
        df_domain = df_results[df_results['domain_uuid'] == domain_uuid].copy()
        calculate_metrics(df_domain, config, dataset, path_to_run_output_dir, domain_uuid, args.n_workers, is_force_refresh=args.is_force_refresh, is_debug=args.is_debug, is_print=False)

if __name__ == "__main__":
    main()