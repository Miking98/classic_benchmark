<div align="center">
  <h1>CLASSic Benchmark</h1>
  <h4>
    <a href="https://huggingface.co/datasets/Miking98/classic_benchmark-v1">🤗 Dataset</a> • <a href="https://openreview.net/forum?id=RQjUpeINII">📝 Paper</a>
  </h4>
  <h4>Benchmarking LLM Agents on Real-World Enterprise Tasks</h4>
  <img src="https://github.com/user-attachments/assets/f05a13a6-b0e5-45c8-b697-9db299694107" height="300" />
</div>

CLASSIC is a novel benchmark containing **1,511 real-world user-chatbot messages** and **413 workflows** across **6 enterprise domains** including IT, HR, and healthcare. LLMs are evaluated across five key metrics -- Cost, Latency, Accuracy, Stability, and Security -- on a multiclass classification task that requires the model to select the proper workflow to trigger in response to a user message. 


### 📖 Table of Contents
1. [Installation](#installation)
1. [Quick Start](#quick_start)
1. [Examples](#examples)
1. [Dataset](#dataset)
1. [Citation](#citation)

<a name="installation"/>

# 💿 Installation

```bash
conda create -n classicbench python=3.10 -y
conda activate classicbench
git clone https://github.com/Miking98/classic_benchmark.git
cd classic_benchmark && pip install -e .
```

<a name="quick_start"/>

# 🚀 Quick Start

Run the benchmark:

```bash
python3 run.py --data [PATH_TO_DATASET_YAML] --agent [PATH_TO_AGENT_YAML]

# Examples:
python3 run.py --data v1 --agent aisera --eval no_security
python3 run.py --data v1 --agent cot_gpt4 --eval no_security
```

Or, download the dataset from [HuggingFace](https://huggingface.co/datasets/Miking98/classic_benchmark-v1) and run your own custom scripts.

```python
from datasets import load_dataset

# Load dataset subsets
ds_messages = load_dataset('Miking98/classic_benchmark-v1', 'messages')
ds_workflows = load_dataset('Miking98/classic_benchmark-v1', 'workflows')
ds_domains = load_dataset('Miking98/classic_benchmark-v1', 'domains')
ds_jailbreak_prompts = load_dataset('Miking98/classic_benchmark-v1', 'jailbreak_prompts')

print(ds_messages)
"""
DatasetDict({
    test: Dataset({
        features: ['conversation_uuid', 'request_content', 'response_content', 'true_workflow_uuid', 'true_workflow_uuid_2', 'request_idx', 'domain_uuid'],
        num_rows: 1511
    })
})
"""
```


<a name="examples"/>

# 👨‍💻 Examples

* Run GPT-4o agent: `python3 run.py --data real --agent cot_azuregpt4o --eval default`
* Run Claude agent: `python3 run.py --data real --agent cot_claude35 --eval default`

<a name="dataset"/>

# 🤗 Dataset

[Download the dataset from 🤗 HuggingFace here](https://huggingface.co/datasets/Miking98/classic_benchmark-v1)

## 📀 Dataset Generation


Listed in order of creation. Each subsequent folder depends on the previous one.

### `./data/0_raw`

Raw data dump from Aisera. 

### `./data/1_sampled`

Next, we sample a subset of chats from the `raw` data dump by running:

```bash
python3 scripts/scripts_to_create_dataset/1_convert_raw_to_sampled.py
```

### `./data/2_annotations`

Next, we generate an Excel file to send to AMT workers to annotate the `sampled` data by running:

```bash
python3 scripts/scripts_to_create_dataset/2_convert_sampled_to_annotations.py
```

IRL, we need to:

1. Use Amazon Mechanical Turk to annotate the chats. Generate one Excel file per annotator.

2. Save the annotated Excel files into `data/2_annotations` and delete the original unannotated Excel file.

### `./data/3_clean`

Next, we clean the dataset from `annotations` by removing conversations flagged by our annotators by running:

```bash
python3 scripts/scripts_to_create_dataset/3_convert_annotations_to_clean.py
```

This is our final, cleaned dataset.

### `./data/4_iclr_workshop_sample.zip`

Submitted to ICLR reviewers.

To generate:

```bash
python3 scripts/scripts_to_create_dataset/4_convert_clean_to_iclr_workshop_sample.py
```

### `./data/5_iclr_workshop_full.zip`

Original dataset reported in ICLR paper.

### `./data/6_hf_dataset`

Convert the dataset to a Hugging Face Dataset and upload it to the Hub.

To generate:

```bash
python3 scripts/scripts_to_create_dataset/6_hf_dataset.py --path_to_dataset_dir ./data/3_clean --hf_version v0
```

<a name="leaderboard" />

# 📊 Leaderboard

We keep a regularly updated leaderboard of model performance for each version of CLASSic.

### v0

* Original dataset from 2025 ICLR Workshop submission.
* Access: *Not released due to privacy considerations.*
* \# of messages: 2311

![all](https://github.com/user-attachments/assets/584d90ee-80cb-44dc-8b97-3df2b60dfacb)

![accuracy](https://github.com/user-attachments/assets/079c8792-e081-4a47-8799-05945ce538e8)

### v1

* Filtered version of **v0**
* Access: [🤗 HuggingFace](https://huggingface.co/datasets/Miking98/classic_benchmark-v1)
* \# of messages: 1511

![all](https://github.com/user-attachments/assets/1074754c-8bb6-49bb-8648-3edd69dc9496)

![accuracy](https://github.com/user-attachments/assets/853b2919-27bb-497d-b17c-7ed075d47adf)


<a name="citation"/>

# 🎓 Citation

```bibtex
@inproceedings{wornow2025top,
  title={Top of the CLASS: Benchmarking LLM Agents on Real-World Enterprise Tasks},
  author={Wornow, Michael and Garodia, Vaishnav and Vassalos, Vasilis and Contractor, Utkarsh},
  booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications}
}
```
