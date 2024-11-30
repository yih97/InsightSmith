
import tinydb
from dotenv import load_dotenv
import os
from modules.formsGenerator.generator_instances import FaceLoraOptions
import json
import logging

load_dotenv()
db = tinydb.TinyDB('/home/hamna/app/VisionForge/tinydb/tinydb.json')

def create_table(table_name: str):
    return db.table(table_name)

def insert_lora(table, data):
    if get(table, tinydb.Query().lora_id == data["lora_id"]):

        raise ValueError("lora_id already exists in database, change lora id and try again.")
    else:
        table.insert(data)
        logging.info(f"Data inserted successfully: {data}")

def insert_bg(table, data):
    if get(table, tinydb.Query().bg_id == data["bg_id"]):
        raise ValueError("bg_id already exists in database, change lora id and try again.")
    else:
        table.insert(data)
        logging.info(f"Data inserted successfully: {data}")

def get(table, query):
    return table.search(query)

if __name__ == "__main__":
    table = create_table("formsGenerator_bg")
    # data = {
    #     "lora_id": "gyj",
    #     "lora_token":"gyj, gyj woman",
    #     "lora_path":"/home/hamna/PycharmProjects/stable-diffusion-webui/models/Lora/gyj.safetensors"
    # }
    #
    # # data = FaceLoraOptions(lora_id="gyj", lora_token="gyj, gyj woman", lora_path="/home/hamna/PycharmProjects/stable-diffusion-webui/models/Lora/gyj.safetensors")
    # # insert(table, data)
    # print(get(table, tinydb.Query().lora_id == "gyj"))


    for item in os.listdir("/home/hamna/app/VisionForge/generator/bg_templates"):
        if item.endswith(".json"):
            lora_id = item.split(".")[0]
            lora_path = os.path.join("/home/hamna/app/VisionForge/generator/bg_templates", item)
            data = {
                "bg_id": lora_id,
                "bg_json": lora_path
            }
            insert_bg(table, data)