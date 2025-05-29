import json
import os
import time
from typing import List, Optional

from pydantic import BaseModel
import litellm
from omegaconf import DictConfig
from classicbench.agents.base import Agent
from classicbench.data.base import Message, PredictionWithMetadata, Prediction, SingleTurnExample, Workflow, Chat
litellm.drop_params = True

class OutputFormat(BaseModel):
    thinking: str
    workflow_name: str

class CoTAgent(Agent):
    """
    Agent that uses CoT to choose a workflow from a list of possible workflows. 
    Supports any model that complies with LiteLLM's `completion` function.
    """

    def __init__(self, config: DictConfig, few_shot_examples: Optional[List[SingleTurnExample]] = None):
        """Initialize the agent."""
        self.config = config
        self.model = config['base_model']
        self.api_base = config.get("api_base", None)
        self.api_version = config.get("api_version", None)
        self.api_key = config.get("api_key", None)
        self.temperature = config.get("temperature", 0.7)
        self.path_to_log_dir = config.get("path_to_log_dir", None)
        self.few_shot_examples = few_shot_examples
        if self.path_to_log_dir:
            os.makedirs(self.path_to_log_dir, exist_ok=True)
    
    def predict(self, example: SingleTurnExample) -> PredictionWithMetadata:
        start = time.time()
        chat: Chat = example.x
        possible_workflows: List[Workflow] = example.possible_y

        # Generate messages
        messages = [
            {
                "role": "system", 
                "content": self.get_system_prompt()
            },
            {
                "role": "user",
                "content": self.get_user_prompt(chat, possible_workflows)
            },
        ]

        max_retries = 4
        wait_time = 60
        data = {}
        result = None
        thinking = ""
        workflow_name = ""
        for i in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    response_format=OutputFormat,
                    max_tokens=512,
                )
                result = response.choices[0].message.content
                data = OutputFormat.model_validate_json(result)
                thinking = data.thinking
                workflow_name = data.workflow_name.strip().lower()
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Rate limited (or other error). Attempt {i+1}/{max_retries}. Waiting for {wait_time}s before retry..")
                time.sleep(wait_time)


        # Map workflow name to a known workflow
        chosen_workflow: Optional[Workflow] = None # default to `None` if no workflow is chosen
        for wf in possible_workflows:
            if wf.name and wf.name.lower() == workflow_name:
                chosen_workflow = wf

        # Create the final prediction
        final_prediction = Prediction(workflow=chosen_workflow, inputs=[], rationale=thinking)

        # Calculate the cost, tokens, and time to predict
        tokens_input: int = litellm.token_counter(messages=messages)
        tokens_output: int = litellm.token_counter(text=result) if result else 0
        cost: float = litellm.completion_cost(completion_response=response) if response else 0
        time_to_pred: float = time.time() - start
        
        # Log everything
        if self.path_to_log_dir:
            with open(os.path.join(self.path_to_log_dir, f"{chat.conversation_uuid}_messages.json"), "w") as f:
                json.dump(messages, f, indent=2)
            with open(os.path.join(self.path_to_log_dir, f"{chat.conversation_uuid}_response.json"), "w") as f:
                json.dump(response.model_dump(), f, indent=2)
            with open(os.path.join(self.path_to_log_dir, f"{chat.conversation_uuid}_parsed_response.json"), "w") as f:
                json.dump(data.model_dump(), f, indent=2)
            with open(os.path.join(self.path_to_log_dir, f"{chat.conversation_uuid}_result.json"), "w") as f:
                json.dump({ 'chosen_workflow': chosen_workflow.to_dict() if chosen_workflow else None, 'final_prediction': final_prediction.to_dict(), 'tokens_input': tokens_input, 'tokens_output': tokens_output, 'cost': cost, 'time_to_pred': time_to_pred }, f, indent=2)

        # Return the prediction with metadata
        return PredictionWithMetadata(
            final_prediction,
            tokens_input,
            tokens_output,
            cost,
            time_to_pred
        )

    def single_turn_example_to_text(self, messages: List[Message]) -> str:
        """Return the chat history as a string."""
        message_strs = []
        for i, m in enumerate(messages):
            if m.sender == 'user':
                message_strs.append(f"- User: {m.content}")
            elif m.sender == 'bot':
                message_strs.append(f"- Bot: {m.content}")
        return '\n'.join(message_strs)
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a helpful assistant that understands a user's request and triggers the appropriate workflow.  You must respond in valid JSON format."""

    def get_user_prompt(self, chat: Chat, workflows: List[Workflow]) -> str:
        """Return the user prompt for the agent which contains the task instructions, chat history, possible workflows, and (optionally) examples."""
        workflow_descriptions = "\n".join([
            f"{idx+1}. {(workflow.name or '').strip()} -- {(workflow.description or 'No description').strip()}" 
            for idx, workflow in enumerate(sorted(workflows, key=lambda x: x.name or ''))
        ])
        
        examples = ""
        if self.few_shot_examples:
            examples = []
            for i, e in enumerate(self.few_shot_examples):
                message_str: str = self.single_turn_example_to_text(e.x.messages)
                examples.append(f"""Example {i+1}:
{message_str}

{{ "thinking": "The workflow {e.y.name} is the best choice because...", "workflow_name": "{e.y.name}" }}""")
            examples = "\n".join(examples)

        prompt = f"""
# Instructions
You are given a chat history with a user, and a list of possible workflows to trigger in response to fulfill the user's most recent request.

# Chat History
Below is the full chat history with the user. The last message is the user's most recent request.
'''
{self.single_turn_example_to_text(chat.messages)}
'''

# Available Workflows
Below is a list of possible workflows that can be triggered in response to the user's most recent request.
The format is `1. name -- description`
When choosing a workflow, make sure the name of the workflow exactly matches one of the names in the list.
{workflow_descriptions}

# Task
Choose the proper workflow to trigger in response to the user's most recent request, or respond with "None" if none of the given workflows are appropriate.

Please respond with ONLY this JSON format and nothing else:
{{
    "thinking" : str <Think step-by-step about which workflow to choose, if any>,
    "workflow_name" : str <The name of the workflow to choose, or "None" if no appropriate workflow can be chosen>
}}

{examples}

IMPORTANT: Your entire response must be a valid JSON object with these two fields only. Do not include any explanations, markdown formatting, or additional text outside the JSON object.""".strip()
        return prompt
    
export = CoTAgent