from typing import List
from omegaconf import DictConfig
import pandas as pd
from classicbench.agents.base import Agent
from classicbench.data.base import PredictionWithMetadata, Prediction, SingleTurnExample, Workflow

class AiseraAgent(Agent):
    """
    Agent that uses Aisera's API to predict the next workflow to call.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        assert config.path_to_labels is not None, "ERROR - config.path_to_labels is not set. Please be sure to set the `path_to_labels` variable in the config to a CSV containing the Aisera workflow labels."
        df_messages = pd.read_csv(config.path_to_labels)
        self.df_labels = df_messages[[
            'aisera_workflow_uuid',
            'aisera_n_tokens',
            'aisera_cost',
            'aisera_latency',
            'conversation_uuid',
            'domain_uuid',
            'request_idx',
        ]].astype({
            'conversation_uuid' : 'str',
            'domain_uuid' : 'str',
            'request_idx' : 'int',
            'aisera_workflow_uuid' : 'str',
            'aisera_n_tokens' : 'int',
            'aisera_cost' : 'float',
            'aisera_latency' : 'float',
        })

    def predict(self, example: SingleTurnExample) -> PredictionWithMetadata:
        """Predict the next workflow to call given the chat history."""
        assert self.df_labels is not None, "ERROR - self.df_labels is not set. Please be sure to set the `path_to_labels` variable in the config to a CSV containing the Aisera workflow labels."
        
        # Get the row in self.df_labels for the conversation and request_idx
        conversation_uuid: str = str(example.x.conversation_uuid) if pd.notna(example.x.conversation_uuid) and example.x.conversation_uuid is not None else None
        request_idx: int = len(example.x.messages) // 2
        try:
            row = self.df_labels[(self.df_labels['conversation_uuid'] == conversation_uuid) & (self.df_labels['request_idx'] == request_idx)].iloc[0]
        except Exception as e:
            print(f"ERROR - No Aisera label found for conversation_uuid={conversation_uuid} and request_idx={request_idx}")
            print(e)
            # No Aisera label found for this conversation and request_idx
            return PredictionWithMetadata(
                Prediction(workflow=None, inputs=[], rationale=None),
                tokens_input=0,
                tokens_output=0,
                cost=None,
                time_to_pred=None,
            )
        
        # Get the actual workflow triggered by Aisera
        workflow_uuid: str = str(row['aisera_workflow_uuid']) if pd.notna(row['aisera_workflow_uuid']) and row['aisera_workflow_uuid'] is not None else None
        domain_uuid: str = str(row['domain_uuid']) if pd.notna(row['domain_uuid']) and row['domain_uuid'] is not None else None
        workflow: Workflow = Workflow(workflow_uuid, domain_uuid, name='', description='', inputs=[], output='')
        
        # Get the actual cost from Aisera
        cost: float = float(row['aisera_cost']) * 1.7 if row['aisera_cost'] is not None and pd.notna(row['aisera_cost']) else None # Adjust for 50% utilization
        
        # Get the actual latency from Aisera
        if row['aisera_latency'] is not None and pd.notna(row['aisera_latency']):
            time_to_pred: float = float(row['aisera_latency']) / 1000 # convert from milliseconds to seconds
        else:
            time_to_pred: float = None
        
        # Get the actual tokens input from Aisera
        tokens_input: int = int(row['aisera_n_tokens']) if pd.notna(row['aisera_n_tokens']) and row['aisera_n_tokens'] is not None else None
        
        # Get the actual tokens output from Aisera
        tokens_output: int = int(row['aisera_n_tokens']) if pd.notna(row['aisera_n_tokens']) and row['aisera_n_tokens'] is not None else None
        
        return PredictionWithMetadata(
            Prediction(workflow, inputs=[], rationale=None),
            tokens_input,
            tokens_output,
            cost,
            time_to_pred,
        )

export = AiseraAgent
