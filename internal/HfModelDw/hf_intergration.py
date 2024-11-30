import logging
import os
import json
from huggingface_hub import hf_hub_download, list_repo_tree, HfApi
from dotenv import load_dotenv
import tinydb
from modules.formsGenerator.database_management import get, create_table, insert_lora,insert_bg,insert_ti
load_dotenv()

# def lora_data(lora_id : str, lora_token : str):
#     data = {
#         "lora_id" :lora_id,
#         "lora_token":lora_token
#             }
#     return data

# def textinversion_data(ti_id : str, ti_token : str, ti_category: str):
#     data = {
#         ti_category:
#             {
#             "ti_id": ti_id,
#             "ti_token": ti_token
#         }
#             }
#     return data
# def mk_db(data, model_category : str):
#     # Load the database and use lora_id
#     if model_category == "Face":
#         table = create_table("Face")
#         insert_lora(table, data)
#
#     elif model_category == "Bg":
#         table = create_table("Bg")
#         insert_bg(table, data)
#
#     elif model_category == "Textinversion":
#         table = create_table("Textinversion")
#         insert_ti(table, data)
#
# def hf_download(model_id : str, model_category : str = "Face", ti_category: str = None) -> str:
#     """
#     Downloads a safetensors file from the Hugging Face Hub and saves it to a specified directory.
#
#     Args:
#         model_id (str): The ID of the Lora model to be downloaded.
#         model_category (str): Model category, Face, Background, TextInversion.
#         ti_category (str, optional): Text Inversion's category.
#
#     Returns:
#         str: The path to the downloaded file.
#
#     Raises:
#         FileExistsError: If the directory already exists.
#
#     """
#     hf_token = os.getenv('HF_Token')
#
#     # db =tinydb.TinyDB(os.getenv("TINYDB_PATH"))
#
#     lora_base_dir = "../../model_zoo/lora"
#     ti_base_dir = "../../model_zoo/TextInversion"
#
#     repo_id = "formsKorea/test_repo"
#
#     model_in_repo = f"{model_category}/{model_id}"
#
#     base_dir = lora_base_dir
#     table_id = "lora_id"
#
#     if model_category == "TextIversion":
#         base_dir = ti_base_dir
#         table_id = "ti_id"
#         if ti_category == "positive":
#             model_in_repo = f"{model_in_repo}/Positive"
#         else:
#             model_in_repo = f"{model_in_repo}/Negative"
#
#     print(model_in_repo)
#     repo_files = list_repo_tree(repo_id=repo_id, path_in_repo=model_in_repo, token=hf_token)
#
#
#     for repo_file in repo_files:
#         repo_file_path = (repo_file.rfilename).split('/')[-1]
#         repo_file_name = repo_file_path.split('.')[0]
#         if repo_file_name == model_id:
#             break
#
#
#     try:
#         new_path = hf_hub_download(repo_id= repo_id, filename=repo_file.rfilename, local_dir=base_dir,
#                                     token=hf_token, repo_type="model")
#
#         # db.update({'model_path': new_path}, tinydb.Query().table_id == model_id )
#
#         logging.info("Successfully downloaded model to %s", os.path.join(base_dir, model_id))
#
#     except FileExistsError:
#         logging.error("Directory already exists: %s", os.path.join(base_dir, model_id))
#
#     return new_path

# def upload_to_huggingface(folder_path : str, user_id : str):
#
#     """
#     Uploads a model folder to the Hugging Face Hub repository.
#     Args:
#         folder_path : 허깅페이스에 업로드할 모델이 있는 경로
#         path_in_repo : 레포안에 모델을 업로드할 디렉토리
#     Returns:
#         None
#     """
#
#     hf_token = os.getenv('HF_Token')
#     repo_id = 'formsKorea/test_repo'
#     api = HfApi()
#     api.upload_folder(
#         folder_path=folder_path,
#         path_in_repo=f"face/{user_id}",
#         repo_id=repo_id,
#         token=hf_token
#     )



if __name__ == "__main__":

    import tinydb
    from modules.formsGenerator.database_management import get, create_table

    # Load the database and use lora_id
    db = tinydb.TinyDB("/home/kyle/kyle/project/VisionForge/internal/HfModelDw/test.json")

    data = lora_data("w2", "dels, dels asian woman")

    mk_db(data=data, model_category="Face")



    FF = '/home/kyle/kyle/project/VisionForge/modules/formsGenerator/test.json'

    with open(FF, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    print(json_data['face'])
    print(json_data['face']['1']['lora_id'])

    db = tinydb.TinyDB("/home/kyle/kyle/project/VisionForge/internal/HfModelDw/test.json")
    file_path = hf_download(json_data['face']['1']['lora_id'], 'Face')

    print(file_path)

    index = db.search(tinydb.Query().Face.lora_id == "w2")[0]
    # table.search(query)
    print(index)

    # db.update({"lora_path": "path"},doc_ids=[index])
    # Download the model

    # data = {
    #     "lora_id": json_data['face']['1']['lora_id'],
    #     "lora_token":"dels, dels woman",
    #     "lora_path": file_path
    # }
    #
    # insert_lora(table,data)

    lora_id = json_data['face']['1']['lora_id']
    json_data['face']['1']['lora_path'] = file_path
