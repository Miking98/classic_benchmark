"""
Given a YAML config, load the corresponding agent's class from the `classicbench/agents/` subfolder.
"""

import importlib
from omegaconf import DictConfig
from classicbench.agents.base import Agent

def load_agent(config: DictConfig) -> Agent:
    """Load agent from `agents/` folder based on config"""
    print(f"Loading agent from {config.agent}")
    try:
        module = importlib.import_module(f"classicbench.agents.{config.agent.name}")
    except Exception as e:
        raise ValueError(f"Error loading agent for `{config.agent.name}`: {e}")
    cls_ = getattr(module, "export")
    agent: Agent = cls_(config.agent)
    return agent