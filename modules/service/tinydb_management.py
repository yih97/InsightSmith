import tinydb
from dotenv import load_dotenv
import os

import logging
load_dotenv()

db =tinydb.TinyDB(os.getenv("TINYDB_PATH"))

def create_table(table_name: str):
    return db.table(table_name)

def data_form(check_point:str, model_category:str, model_id:str, token:str,  ti_category: str = None, model_path : str = None) -> dict:
    """
    Creates a dictionary with model configuration data based on the model category.

    Args:
        check_point (str): The checkpoint value for the model.
        model_category (str): The category of the model. It can be "FACE", "BG", or "TI".
        model_id (str): The identifier for the model.
        token (str): The token used for authentication.
        ti_category (str, optional): The category for text inversion models. Default is None. positive or negative

    Returns:
        dict: A dictionary containing the model configuration data.

    Raises:
        ValueError: If the model_category is not "FACE", "BG", or "TI".

    Example:
         ex_data= data_form(check_point="majicmix",model_category="FACE",model_id="w2",token="dels, dels asian woman")
    """

    face_path = os.getenv("FACELORA_PATH")
    bg_path = os.getenv("BGLORA_PATH")
    ti_path = os.getenv("TEXTINVERSION_PATH")


    if model_category == "FACE":
        if model_path:
            data = {
                "check_point":check_point,
                "model_category":model_category,
                "model_id":model_id,
                "lora_token":token,
                "lora_path":model_path
            }
        else:
            data = {
                "check_point": check_point,
                "model_category": model_category,
                "model_id": model_id,
                "lora_token": token,
                "lora_path": face_path
            }

    elif model_category == "BG":
        data = {
            "check_point": check_point,
            "model_category": model_category,
            "model_id": model_id,
            "lora_token": token,
            "lora_path" : bg_path
        }

    elif model_category == "TI":
        data = {
            "check_point": check_point,
            "model_category": model_category,
            "model_id": model_id,
            "ti_token": token,
            "ti_category":ti_category,
            "ti_path" : (f"{ti_path}/{ti_category}")
        }

    else:
        logging.info("The model_category is invalid. It must be one of FACE, BG, or TI.")

    logging.info(f"created data form, model_category : {model_category}")
    return data

def insert_data(table, data : dict):
    """
     Inserts or updates data in the given table.

     Args:
         table (tinydb.Table): The TinyDB table where data is to be inserted or updated.
         data (dict): A dictionary containing the model configuration data.
                      It must include the keys "model_category" and an identifier key
                      which is either "user_id", "ti_id", or "bg_id" depending on the model_category.

     Raises:
         ValueError: If the model_category is not "FACE", "TI", or "BG".
                     If the identifier_key already exists in the database.

     Example:
        insert_data(table,data0)
     """

    if data["model_category"] in ["FACE", "BG", "TI"]:
        existing_data = get(table, (tinydb.Query().model_category == data["model_category"]))
        if existing_data:
            if get(table, (tinydb.Query()["model_id"] == data["model_id"])):
                raise logging.info(ValueError(f"{data['model_id']}already exists in database, change lora id and try again."))
            else:
                table.update(data)
                logging.info(f"Data updated successfully: {data}")
        else:
            table.insert(data)
    else:
        raise logging.info(ValueError("Invalid model_category. Must be 'FACE', 'TI', or 'BG'."))



def load_setting(table, id):
    results = table.search(tinydb.Query().model_id == id)[0]
    return results
def get(table, query):
    return table.search(query)


if __name__ == "__main__":

    # table = create_table("snapform")
    table=db.table("snapform")

    data0 = data_form(check_point="majicmix", model_category="FACE", model_id="w2", token="dels, dels asian woman")
    data1 = data_form(check_point="majicmix", model_category="BG", model_id="city", token="casca, ascasc city")
    data2 = data_form(check_point="majicmix", model_category="TI", model_id="baddream", token="BADDREAM", ti_category="positive")

    insert_data(table,data0)
    insert_data(table,data1)
    insert_data(table,data2)

    a= load_setting(table,"w2")
    print(a)
