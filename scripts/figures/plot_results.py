"""
Usage:
    python3 plot_results.py

Purpose:
    Plot results from runs across all models / metrics.
"""
import argparse
import json
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import os
from classicbench.utils import get_rel_path
import seaborn as sns
sns.set_theme(style="whitegrid")

# Plot styling
MODELS = ["geminipro15", "gpt4o", "claude35", "aisera", "o3mini"]
METRICS = ["cost", "latency", "accuracy", "stability", "security"]
METRIC_TO_LABEL = {
    "cost": "Cost ($)",
    "latency": "Latency (s)",
    "accuracy": "Accuracy (pass@1)",
    "stability": "Stability (pass^2)",
    "security": "Security (%)",
}
MODEL_TO_LABEL = {
    "geminipro15": "Gemini 1.5 Pro",
    "gpt4o": "GPT-4o",
    "claude35": "Claude 3.5 Sonnet",
    "aisera": "Aisera",
    "o3mini": "o3-mini",
}
DOMAIN_TO_LABEL = {
    "finance": "FinTech",
    "medical": "Medical",
    "hr": "HR",
    "it": "IT",
    "edtech": "EdTech",
    "biotech": "Biotech",
    "bank": "Banking",
    "all": "Overall",
}
MODEL_TO_COLOR = {
    "geminipro15": "lightblue",
    "gpt4o": "lightgreen",
    "claude35": "plum",
    "aisera": "lightcoral",
    "o3mini": "lightgray",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data_dir", type=str, default=get_rel_path("./data/3_clean_v0/"))
    parser.add_argument("--path_to_results_dir", type=str, default=get_rel_path("./results_v0/"))
    parser.add_argument("--path_to_output_dir", type=str, default=get_rel_path("./figures_v0/"))
    return parser.parse_args()

def plot_all_metrics(df: pd.DataFrame, 
                    model_to_color: Dict[str, str], 
                    suptitle: str = 'CLASSIC Results',
                    path_to_save: str = None):
    """Generate a plot containing 5 subplots, each with a bar chart of a different metric (cost, latency, accuracy, stability, security).
    """
    # Creating subplots for each category with no y-axis labels and lighter custom colors
    fig, axes = plt.subplots(1, df['metric'].nunique(), figsize=(20, 6), sharey=False)

    for i, metric in enumerate(df['metric'].unique()):
        df_metric = df[df["metric"] == metric]
        ax = axes[i]
        raw_values = df_metric["value"]
        model_colors = [model_to_color[model] for model in df_metric["model"]]
        bars = ax.bar(df_metric["model"], raw_values, color=model_colors)
        ax.set_title(METRIC_TO_LABEL[metric])
        ax.set_xticks(range(len(df_metric["model"])))
        ax.set_xticklabels([MODEL_TO_LABEL[model] for model in df_metric["model"]], rotation=45, ha='right')
        
        # Annotate each bar with its raw value
        for bar, value in zip(bars, raw_values):
            if metric == 'latency':
                metric_fmt = f"{value:.1f}"
            else:
                metric_fmt = f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.02 * max(raw_values)), 
                    metric_fmt, ha='center', va='bottom')
        
        # Extend y-axis limits for better spacing
        ax.set_ylim(0, max(raw_values) * 1.2)

    # Adjust layout for better spacing
    fig.suptitle(suptitle)
    plt.tight_layout()
    # plt.show()
    if path_to_save:
        plt.savefig(path_to_save)

def plot_one_metric_by_domain(df: pd.DataFrame, metric: str, model_to_color: Dict[str, str], suptitle: str = 'CLASSIC Results', path_to_save: str = None):
    """Generate a plot containing 6 subplots (one for each domain) with a bar chart of a specific metric
    """
    # Creating subplots for each domain with lighter custom colors
    fig, axes = plt.subplots(1, df['domain'].nunique(), figsize=(25, 6), sharey=False)

    y_lim_max = df["value"].max()
    for i, domain in enumerate(df['domain'].unique()):
        df_domain = df[df['domain'] == domain]
        ax = axes[i]
        raw_values = df_domain["value"]
        model_colors = [model_to_color[model] for model in df_domain["model"]]
        bars = ax.bar(df_domain["model"], raw_values, color=model_colors)
        ax.set_title(DOMAIN_TO_LABEL[domain])
        ax.set_xticks(range(len(df_domain["model"])))
        ax.set_xticklabels([MODEL_TO_LABEL[model] for model in df_domain["model"]], rotation=45, ha='right')
        
        # Annotate each bar with its raw value
        for bar, value in zip(bars, raw_values):
            metric_fmt = f"{value:.1f}%" if metric == "latency" else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.02 * max(raw_values)), 
                    metric_fmt, ha='center', va='bottom')
        
        # Extend y-axis limits for better spacing
        ax.set_ylim(0, y_lim_max * 1.2)

    # Adjust layout for better spacing
    plt.suptitle(suptitle)
    plt.tight_layout()
    # plt.show()
    if path_to_save:
        plt.savefig(path_to_save)

######################
# Results
######################

def parse_overall_metrics_json_to_df(path_to_json: str, model: str) -> pd.DataFrame:
    data = json.load(open(path_to_json))
    rows = []
    for domain in DOMAIN_TO_LABEL.keys():
        for (metric, metric_key_name) in [
            ("cost", "cost_mean"),
            ("latency", "time_to_pred_mean"),
            ("accuracy", "row_pass@1"),
            ("stability", "row_pass^2"),
            ("security", "security"),
        ]:
            if (metric not in data or data[metric] is None or (isinstance(data[metric], dict) and f'domain:{domain}' not in data[metric])):
                # Don't have results for this domain/metric, so skip
                continue
            if metric == "security":
                rows.append([model, domain, metric, data[metric]])
            else:
                assert f'domain:{domain}' in data[metric], f"Metric `{metric}` does not have a `domain:{domain}` key."
                rows.append([model, domain, metric, data[metric][f'domain:{domain}'][metric_key_name]])
    df = pd.DataFrame(rows, columns=["model", "domain", "metric", "value"])
    return df

if __name__ == "__main__":
    args = parse_args()
    path_to_data_dir = args.path_to_data_dir
    path_to_results_dir = args.path_to_results_dir
    path_to_output_dir = args.path_to_output_dir
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load raw datset
    df_messages = pd.read_csv(os.path.join(path_to_data_dir, "messages.csv"))
    df_domains = pd.read_csv(os.path.join(path_to_data_dir, "domains.csv"))
    df_workflows = pd.read_csv(os.path.join(path_to_data_dir, "workflows.csv"))
    
    # Calculate # of messages per domain
    domain_2_msg_count: Dict[str, int] = df_messages['domain_uuid'].value_counts().to_dict()
    domain_2_msg_count['all'] = sum(domain_2_msg_count.values())

    # Calculate mean/std of results across runs within each model
    mean_dfs = []
    std_dfs = []
    for model in os.listdir(path_to_results_dir):
        if model not in MODELS:
            print(f"Skipping `{model}` because it's not in `MODELS`.")
            continue
        if not os.path.isdir(f"{path_to_results_dir}/{model}"):
            print(f"Skipping `{model}` because it doesn't have any results.")
            continue
        sub_dfs = []
        for run_name in os.listdir(f"{path_to_results_dir}/{model}"):
            if not os.path.isdir(f"{path_to_results_dir}/{model}/{run_name}"):
                continue
            sub_dfs.append(parse_overall_metrics_json_to_df(f"{path_to_results_dir}/{model}/{run_name}/overall_metrics.json", model))
        mean_sub_df = pd.concat(sub_dfs).groupby(["model", "domain", "metric"]).mean().reset_index()
        mean_dfs.append(mean_sub_df)
        if len(sub_dfs) > 1:
            std_sub_df = pd.concat(sub_dfs).groupby(["model", "domain", "metric"]).std().reset_index()
            std_dfs.append(std_sub_df)
        else:
            print(f"Only one run found for `{model}`. Skipping std calculation. This is expected for the 'aisera' model, but not for other models.")
    mean_df = pd.concat(mean_dfs)
    std_df = pd.concat(std_dfs)

    # Reorder models in order of [gemini, gpt4, claude, aisera]
    mean_df["model"] = pd.Categorical(mean_df["model"], categories=MODELS, ordered=True)
    std_df["model"] = pd.Categorical(std_df["model"], categories=MODELS, ordered=True)
    # Reorder metrics in order of [cost, latency, accuracy, stability, security]
    mean_df["metric"] = pd.Categorical(mean_df["metric"], categories=METRICS, ordered=True)
    std_df["metric"] = pd.Categorical(std_df["metric"], categories=METRICS, ordered=True)
    mean_df = mean_df.sort_values(by=["model", "metric", "domain"])
    std_df = std_df.sort_values(by=["model", "metric", "domain"])

    # Join (mean_df, std_df) => df with columns: model, metric, domain, value_mean, value_std
    df = pd.merge(mean_df, std_df, on=["model", "metric", "domain"], suffixes=("_mean", "_std"), how="outer")
    df["value_fmt"] = df.apply(lambda row: f"{row['value_mean']:.3f} ± {row['value_std']:.3f}" if pd.notna(row['value_std']) else f"{row['value_mean']:.3f} ± NaN", axis=1)

    # Adjusting the "Cost ($)" category to index at 1 for Aisera
    # for domain in df["domain"].unique():
    #     cost_indexed = df.loc[ (df['domain'] == domain) & (df["metric"] == "cost"), "value" ].values / df.loc[(df['domain'] == domain) & (df["metric"] == "cost") & (df["model"] == "aisera"), "value"].values[0]
    #     df.loc[(df["domain"] == domain) & (df['metric'] == "cost"), "value"] = cost_indexed
    # assert (df.loc[(df['model'] == "aisera") & (df['metric'] == "cost"), "value"].values == 1.0).all() # all Aisera costs must be 1

    ######################
    # Make table of results
    ######################
    for domain in df["domain"].unique():
        df_domain = df[df["domain"] == domain]
        df_pivot = df_domain.pivot(index="model", columns="metric", values="value_fmt").reset_index()
        df_pivot['model'] = df_pivot['model'].map(MODEL_TO_LABEL)
        if domain != 'all':
            # Drop security column
            if 'security' in df_pivot.columns:
                df_pivot = df_pivot.drop(columns=["security"])
        df_pivot.to_csv(f"{path_to_output_dir}/table_{domain}.csv", index=False)
        df_pivot.to_latex(f"{path_to_output_dir}/table_{domain}.tex", index=False)

    # Make table of just accuracy
    df_accuracy = df[df["metric"] == "accuracy"]
    df_accuracy_pivot = df_accuracy.pivot(index="model", columns="domain", values="value_fmt").reset_index()
    # Order columns by domain_2_msg_count.keys()
    df_accuracy_pivot = df_accuracy_pivot[['model'] + list(domain_2_msg_count.keys())]
    df_accuracy_pivot['model'] = df_accuracy_pivot['model'].map(MODEL_TO_LABEL)
    df_accuracy_pivot.to_csv(f"{path_to_output_dir}/table_accuracy.csv", index=False)
    df_accuracy_pivot.to_latex(f"{path_to_output_dir}/table_accuracy.tex", index=False)

    ######################
    # Make plots of results
    ######################
    df['value'] = df['value_mean']

    # Plot results by domain (NOTE: This includes 'overall' domain)
    for domain in df["domain"].unique():
        df_domain = df[df["domain"] == domain]
        if domain != "overall":
            # Ignore security metric for non-overall domains
            df_domain = df_domain[df_domain['metric'] != 'security']
        plot_all_metrics(df_domain, MODEL_TO_COLOR, suptitle=f"{DOMAIN_TO_LABEL[domain]} Results", path_to_save=f"{path_to_output_dir}/{domain}.png")

    # Plot just accuracy
    df_accuracy = df[df["metric"] == "accuracy"]
    plot_one_metric_by_domain(df_accuracy, "accuracy", MODEL_TO_COLOR, suptitle="Accuracy Results", path_to_save=f"{path_to_output_dir}/accuracy.png")

    ######################
    # Make plots of dataset
    ######################

    # Make horizontal bar plot of number of messages per domain
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([DOMAIN_TO_LABEL[x] for x in df_messages["domain_uuid"].value_counts().index], 
                df_messages["domain_uuid"].value_counts())
    ax.set_title("Number of messages per domain")
    ax.set_xlabel("Number of messages")
    ax.set_ylabel("Domain")
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}',  # Format number with commas
                ha='left', va='center', fontsize=10)
    plt.margins(x=0.1)
    plt.savefig(f"{path_to_output_dir}/messages_per_domain.png", bbox_inches='tight')

    # Make bar plot of number of messages per conversation
    fig, ax = plt.subplots(figsize=(10, 6))
    conversation_lengths = df_messages['conversation_uuid'].value_counts()
    sns.histplot(conversation_lengths, bins=20, kde=False, ax=ax, color='lightblue')
    ax.set_title("Number of messages per conversation")
    ax.set_xlabel("Number of messages")
    ax.set_ylabel("Number of conversations")
    plt.savefig(f"{path_to_output_dir}/messages_per_conversation.png", bbox_inches='tight')