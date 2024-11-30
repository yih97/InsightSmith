import tinydb
from dotenv import load_dotenv
import os

from huggingface_hub import hf_hub_download, list_repo_tree, HfApi

import logging
load_dotenv()
db =tinydb.TinyDB(os.getenv("TINYDB_PATH"))
table=db.table("snapform")
hf_token = os.getenv('HF_Token')

def create_table(table_name: str):
    return db.table(table_name)

def data_form(model_category:str, model_id:str, token:str) -> dict:
    """
    Creates a dictionary with model configuration data based on the model category.

    Args:
        model_category (str): The category of the model. It can be "FACE", "BG", or "TI".
        model_repo (str) : The repo path in huggingface
        model_id (str): The identifier for the model.
        token (str): The token used for authentication.


    Returns:
        dict: A dictionary containing the model configuration data.

    Raises:
        ValueError: If the model_category is not "FACE", "BG", or "TI".

    Example:
         ex_data= data_form(check_point="majicmix",model_category="FACE",model_id="w2",token="dels, dels asian woman")
    """
    model_repo = f"{model_category}/{model_id}.safetensors"

    if model_category == "FACE":
        os.getenv("FACELORA_PATH") + f"{model_id}.safetenosrs"
        lora_path = os.path.join(os.environ.get('FACELORA_PATH'),
                                 model_id + ".safetensors")
        data = {
            "model_category": model_category,
            "model_id": model_id,
            "token": token,
            "model_path": lora_path,
            "model_repo": model_repo
        }

    elif model_category == "BG":
        lora_path = os.path.join(os.environ.get('BGLORA_PATH'),
                                 model_id + ".safetensors")

        data = {
            "model_category": model_category,
            "model_id": model_id,
            "token": token,
            "model_path" : lora_path,
            "model_repo": model_repo
        }

    elif model_category == "TI":

        model_repo = f"{model_category}/{model_id}.pt"
        # ti_path = os.path.join(os.environ.get('TEXTINVERSION_PATH'),
        #                          model_id.pt)
        data = {
            "model_category": model_category,
            "model_id": model_id,
            "token": token,
            "model_path" : "../../model_zoo/TI/",
            "model_repo": model_repo
        }

    else:

        logging.info("The model_category is invalid. It must be one of FACE, BG, or TI.")

    logging.info(f"created data form, model_category : {model_category}")
    return data

def update_data(model_id, data : dict):
    """
    Args:
        table (tinydb.Table): The TinyDB table where data is to be inserted or updated.
        data (dict) : The data to be inserted or updated.
    :return:
    """
    if get(table, (tinydb.Query().model_id == model_id)):
        table.update(data)
    else:
        logging.error("the model_id is not exist in DB")
        raise ValueError("the model_id is not exist in DB")


def insert_data(data: dict):
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

        existing_data = get(table, tinydb.Query().model_category==["FACE", "BG", "TI"])
        if existing_data:
            if get(table, (tinydb.Query()["model_id"] == data["model_id"])):
                logging.error(f"{data['model_id']} already exists in database, change model id and try again.")
                raise ValueError(f"{data['model_id']} already exists in database, change model id and try again.")
            else:
                table.update(data)
                logging.info(f"Data updated successfully: {data}")
        else:
            table.insert(data)
    else:
        logging.error(ValueError("Invalid model_category. Must be 'FACE', 'TI', or 'BG'."))
        raise ValueError("Invalid model_category. Must be 'FACE', 'TI', or 'BG'.")

def load_setting(table, id):
    try:
        logging.info('Checking if user lora info exists in DB')
        result = table.search(tinydb.Query().model_id==id)[0]

        if result:
            return result
    except:
        error_msg = "lora info not exist in DB"
        logging.info(error_msg)
        raise ValueError(error_msg)


def get(table, query):
    return table.search(query)

def exist_lora(id):
    # add google doc string tomorrow

    table = db.table("snapform")
    result = table.search(tinydb.Query().model_id == id)
    repo_id = "formsKorea/test_repo"

    if result[0]:
        repo_files = list_repo_tree(repo_id=repo_id, path_in_repo="Face/w2", token=hf_token)
        try:
            for repo_file in repo_files:
                print(repo_file.rfilename)
                logging.info("seach huggingface repo")
                repo_file_path = (repo_file.rfilename).split('/')[-1]
                repo_file_name = repo_file_path.split('.')[0]
                print(repo_file_name)
                if repo_file_name == "w2":
                    break

            lora_path = hf_hub_download(repo_id=repo_id, filename=repo_file.rfilename, local_dir= os.getenv("FACELORA_PATH"),
                                       token=hf_token, repo_type="model")
        except:
            logging.info("model is not exist in repo")
            raise ValueError("model is not exist in repo")

    else:
        logging.info("model is not exist in repo path")
        raise ValueError("model is not exist in repo path")
    exist = True
    return exist ,lora_path

def exist_ti():
    # add google doc string tomorrow

    hf_token = os.getenv('HF_Token')
    table = db.table("snapform")
    ti_data = table.search(tinydb.Query().model_category == "TI")
    repo_id = "formsKorea/test_repo"

    ti_path_list = []

    if ti_data:
        repo_files = list_repo_tree(repo_id=repo_id, path_in_repo="TI", token=hf_token)
        try:
            for i,repo_file in enumerate(repo_files):
                logging.info("search huggingface repo")
                repo_file_path = (repo_file.rfilename).split('/')[-1]
                repo_file_name = repo_file_path.split('.')[0]
                if repo_file_name == ti_data[i]["model_id"]:
                    ti_path = hf_hub_download(repo_id=repo_id, filename=repo_file.rfilename,
                                              local_dir=os.getenv("TEXTINVERSION_PATH"),
                                              token=hf_token, repo_type="model")
                    ti_path_list.append(ti_path)
        except Exception as e:
            logging.info(f"error raise {e}")
            raise ValueError(f"error raise {e}")
    else:
        logging.info("model is not ti_data in repo path")
        raise ValueError("model is not ti_data in repo path")

    return ti_path_list

# def check_lora_exists(uuid: str):
    # i used only support test
#     try:
#         logging.info('Checking if user lora exists')
#         # if gender.lower() == 'f':
#         #     select_epoch = os.environ.get('EPOCH_NUMBER_F')
#         # else:
#         #     select_epoch = os.environ.get('EPOCH_NUMBER_M')
#
#         lora_path = os.path.join(os.environ.get('Lora_Paths'),
#                                  uuid + ".safetensors")
#         exists = os.path.exists(lora_path)
#         print(exists)
#         if exists == None:
#             return exists, lora_path
#         else:
#             lora_path = exist_lora(uuid)
#
#             return exists, lora_path
#
#     except Exception as e:
#         logging.error(f"Error checking if user lora exists: {e}")
#         raise e

if __name__ == "__main__":
    #import glob
    from pathlib import Path

    hf_token = os.getenv('HF_Token')
    repo_id = "formsKorea/test_repo"

    # table = create_table("snapform")
    table=db.table("snapform")

    data0 = data_form(model_category="FACE", model_id="w2", token="dels, dels asian woman")
    data1 = data_form(model_category="BG", model_id="city", token="casca, ascasc city")
    data2 = data_form(model_category="TI", model_id="BadDream", token="BadDream")
    data3 = data_form(model_category="TI", model_id="UnrealisticDream", token="UnrealisticDream")


    insert_data(data0)
    insert_data(data1)
    insert_data(data2)
    insert_data(data3)

    y= exist_lora("w2")
    print(f"??{y}")

    a = exist_ti()
    print(a)


    # I only use test
    # updat_data(table,"w2", {"model_repo":"./model/zoo"})
    # a = load_setting(table,"w1")
    # print(a['model_id'])
    # exists, lora_path = check_lora_exists("w2")
    #
    # print(f"{exists}, {lora_path}")

    # try:
    #     text_inversion_path_list = glob.glob(os.path.join("/home/kyle/kyle/project/VisionForge/model_zoo/Textinversion/TI", '*.pt'))
    #     if len(text_inversion_path_list) == 0:
    #         text_inversion_path_list = exist_ti()
    #     token = []
    #     for text_inversion in text_inversion_path_list:
    #         text_inversion_name = Path(text_inversion).stem
    #         token.append(text_inversion_name)
    #         print(text_inversion_name)
    #     #     self.checkpoint.load_textual_inversion(text_inversion, token=token)
    #     #
    #     # textual_inversion_manager = DiffusersTextualInversionManager(self.checkpoint)
    #     # self.compel = Compel(tokenizer=self.checkpoint.tokenizer, text_encoder=self.checkpoint.text_encoder,
    #     #                      textual_inversion_manager=textual_inversion_manager)
    #     logging.info("Textual inversion loaded successfully")
    # except Exception as e:
    #     logging.error(f"Error while loading textual inversion: {e}")