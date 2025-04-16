# Top of the CLASS

Benchmarking LLM Agents on Real-World Enterprise Tasks

# 📖 Table of Contents
1. [Installation](#installation)
1. [Quick Start](#quick_start)
1. [Examples](#examples)
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

1. Download the dataset [TODO]

2. Run the benchmark:

```bash
python3 run.py --data [PATH_TO_DATASET_YAML] --agent [PATH_TO_AGENT_YAML]
```

<a name="examples"/>

# 👨‍💻 Examples

* Run GPT-4o agent on CLASSIC benchmark: `python3 run.py --data real --agent cot_azuregpt4o --eval default`
* Run Claude agent on CLASSIC benchmark: `python3 run.py --data real --agent cot_claude35 --eval default`

# 🎓 Citation

```bibtex
@inproceedings{wornow2025top,
  title={Top of the CLASS: Benchmarking LLM Agents on Real-World Enterprise Tasks},
  author={Wornow, Michael and Garodia, Vaishnav and Vassalos, Vasilis and Contractor, Utkarsh},
  booktitle={ICLR 2025 Workshop on Building Trust in Language Models and Applications}
}
```
