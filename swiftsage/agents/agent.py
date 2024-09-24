from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, prompt_template, llm_client):
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    @abstractmethod
    def generate_response(self, prompt):
        pass