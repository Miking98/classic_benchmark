<div align="center">
  <h1>CLASSic Benchmark</h1>
  <h4>
    <a href="https://huggingface.co/datasets/Miking98/classic_benchmark">🤗 Dataset</a> • <a href="https://openreview.net/forum?id=RQjUpeINII">📝 Paper</a>
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
conda create -n classic python=3.10 -y
conda activate classic
git clone https://github.com/Miking98/classic_benchmark.git
cd classic_benchmark && pip install -e .
```

<a name="quick_start"/>

# 🚀 Quick Start

1. Run the benchmark:

```bash
python3 run.py --data [PATH_TO_DATASET_YAML] --agent [PATH_TO_AGENT_YAML]
```

2. Or, download the dataset from [HuggingFace](https://huggingface.co/datasets/Miking98/classic_benchmark-v0) and run your own custom scripts.

```python
from datasets import load_dataset

# Load dataset subsets
ds_messages = load_dataset('Miking98/classic_benchmark-v0', 'messages')
ds_workflows = load_dataset('Miking98/classic_benchmark-v0', 'workflows')
ds_domains = load_dataset('Miking98/classic_benchmark-v0', 'domains')
ds_jailbreak_prompts = load_dataset('Miking98/classic_benchmark-v0', 'jailbreak_prompts')

print(ds_messages)
"""
DatasetDict({
    test: Dataset({
        features: ['conversation_uuid', 'request_content', 'response_content', 'true_workflow_uuid', 'true_workflow_uuid_2', 'aisera_workflow_uuid', 'aisera_n_tokens', 'aisera_cost', 'aisera_latency', 'request_idx', 'domain_uuid'],
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

[Download the dataset from HuggingFace here](https://huggingface.co/datasets/Miking98/classic_benchmark)


<a name="citation"/>

# 🎓 Citation

```bibtex
@inproceedings{wornow2025top,
  title={Top of the CLASS: Benchmarking LLM Agents on Real-World Enterprise Tasks},
  author={Wornow, Michael and Garodia, Vaishnav and Vassalos, Vasilis and Contractor, Utkarsh},
  booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications}
}
```
