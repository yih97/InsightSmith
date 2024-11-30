
import wandb
import logging
import os

from dotenv import load_dotenv

class logger:
    def __init__(self, project_name: str):
        load_dotenv()
        wandb_token = os.getenv('WANDB_TOKEN')
        entity = os.getenv('WANDB_ENTITY')
        project_name = os.getenv('WANDB_PROJECT')
        wandb.init(project=project_name, entity=entity)
        self.log_data = []

    def log(self, data: dict):
        self.log_data.append(data)
        wandb.log(data)

# class logger:
#     class logger:
#         def __init__(self):
#             load_dotenv()
#             wandb_token = os.getenv('WANDB_TOKEN')
#             if wandb_token is None:
#                 raise ValueError("WANDB_TOKEN not found in .env file. Please make sure it's defined.")
#             entity = os.getenv('WANDB_ENTITY')
#             if entity is None:
#                 raise ValueError("WANDB_ENTITY not found in .env file. Please make sure it's defined.")
#             self.project_name = os.getenv('WANDB_PROJECT')
#             if self.project_name is None:
#                 raise ValueError("WANDB_PROJECT not found in .env file. Please make sure it's defined.")
#             self.log_data = []
#
#         def initialize(self):  # initialize 메서드 추가
#             wandb.login(key=wandb_token)
#             wandb.init(project=self.project_name, entity=entity)
#
#         def log(self, data: dict):
#             self.log_data.append(data)
#             wandb.log(data)