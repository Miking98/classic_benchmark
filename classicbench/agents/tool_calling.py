import json
import re
import time
from litellm import completion, completion_cost, token_counter
from typing import List, Dict, Any
import litellm
from omegaconf import DictConfig
from classicbench.data.base import Prediction, PredictionWithMetadata, SingleTurnExample, Workflow, Chat

class ToolCallingAgent:
    def __init__(self, config: DictConfig):
        self.config = config
        self.name_fallback_counter = 0
        self.model = config['base_model']
        self.api_base = config.get("api_base", None)
        self.api_version = config.get("api_version", None)
        self.api_key = config.get("api_key", None)
        self.temperature = config.get("temperature", 0.7)
        self.max_turns: int = config.get("max_turns", 1)
        assert litellm.supports_function_calling(model=self.model) == True, "Model must support function calling!"

    def format_chat_messages(self, chat: Chat) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in chat.messages:
            if message.content.strip() == '':
                continue
            role = 'assistant' if message.sender == 'bot' else 'user'
            formatted_messages.append({"role": role, "content": message.content})
        return formatted_messages

    def predict(self, example: SingleTurnExample) -> PredictionWithMetadata:
        start_time = time.time()
        self.name_fallback_counter = 0
        chat: Chat = example.x
        possible_workflows: List[Workflow] = example.possible_y
        formatted_messages = self.format_chat_messages(chat)
        prompt = self.get_agent_prompt(possible_workflows)
        formatted_messages.insert(0, {"role": "system", "content": prompt})
        tools = [
            {
                "type": "function",
                "function": {
                    "name": self.sanitize_tool_name(workflow.name),
                    "description": workflow.description,
                },
            }
            for workflow in possible_workflows
        ]
        if len(tools)>128:
            print("Error: Tool calling agent does not support more than 128 tools")
            return PredictionWithMetadata(
            prediction=None,
            tokens_input=0,
            tokens_output=0,
            cost=0,
            time_to_pred=0
        )
        response = None
        retries = 0
        max_attempts = 2
        wait_time = 60
        for turn_index in range(self.max_turns):
            while retries < max_attempts:
                try:
                    retries += 1

                    response = completion(
                        model=self.model,
                        api_base=self.api_base,
                        api_version=self.api_version,
                        api_key=self.api_key,
                        messages=formatted_messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=self.temperature,
                    )
                    break
                except Exception as e:
                    print(f"Encountered error: {e}. Attempt {retries}/{max_attempts}.")
                    if retries < max_attempts:
                        print(f"Waiting for {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print("Exceeded maximum retries. No valid response from LLM.")
            if response is not None:
                break
        if response is None:
            tokens_input = token_counter(messages=formatted_messages)
            tokens_output = 0
            cost = 0.0
            time_to_pred = time.time() - start_time
            print("Warning: No LLM response. Returning None workflow.")
            return PredictionWithMetadata(
                prediction=None,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                time_to_pred=time_to_pred
            )
        chosen_predictions = []
        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                for workflow in possible_workflows:
                    if self.sanitize_tool_name(tool_call.function.name) == self.sanitize_tool_name(workflow.name):
                        inputs = tool_call.function.arguments
                        prediction = Prediction(workflow=workflow, inputs=inputs, rationale=None)
                        chosen_predictions.append(prediction)
                        break
        tokens_input = token_counter(messages=formatted_messages)
        tokens_output = token_counter(text=response_message.content if response_message.content else "")
        cost = completion_cost(completion_response=response)
        time_to_pred = time.time() - start_time
        return PredictionWithMetadata(
            prediction=chosen_predictions[-1] if chosen_predictions else None,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            time_to_pred=time_to_pred
        )

    def get_agent_prompt(self, workflows: List[Workflow]) -> str:
        prompt = f"""
        You are a tool-calling agent that determines the appropriate workflow to execute based on the conversation context.
        When you decide to call a workflow, you will emit a function-call with the relevant arguments.
        """.strip()
        return prompt
    
    def sanitize_tool_name(self, name: str) -> str:
        """
        Sanitize a workflow/tool name to meet Azure's '^[a-zA-Z0-9_-]+$' pattern.
        Ensure uniqueness by tracking used names and appending a suffix if there's a collision.
        """
        try:
            # Replace invalid chars with '_'
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            if not sanitized:
                sanitized = f"No_name_{self.name_fallback_counter}"
                self.name_fallback_counter += 1
            if not re.match(r'^[a-zA-Z_]', sanitized):
                sanitized = f"_{sanitized}"
            if self.model == "gemini/gemini-1.5-pro":
                sanitized = sanitized[:63]
            else:
                sanitized = sanitized[:64]
        except:
            sanitized = f"No_name_{self.name_fallback_counter}"
            self.name_fallback_counter += 1
        return sanitized

export = ToolCallingAgent
