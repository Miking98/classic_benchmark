import time
from typing import List

from omegaconf import DictConfig
from classicbench.agents.base import Agent
from classicbench.data.base import PredictionWithMetadata, SingleTurnExample, Workflow

class DefaultAgent(Agent):
    """Agent that predicts the next workflow to call"""
    
    def __init__(self, config: DictConfig):
        pass
    
    def predict(self, example: SingleTurnExample) -> PredictionWithMetadata:
        """Predict the next workflow to call given the chat history."""
        start = time.time()
        time_to_pred: float = time.time() - start
        possible_workflows: List[Workflow] = example.possible_y
        return PredictionWithMetadata([possible_workflows[0]], 0, 0, 0, time_to_pred)

export = DefaultAgent