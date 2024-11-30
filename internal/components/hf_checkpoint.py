from huggingface_hub import create_repo, HfApi, upload_folder
from diffusers import StableDiffusionPipeline
import os
import logging
from dotenv import load_dotenv

load_dotenv()


def upload_sd_to_huggingface(repo_name, model_path, path_in_repo):
    """
    Uploads a local model to Hugging Face Hub. https://huggingface.co/formsKorea

    Example:
    ```
    upload_to_huggingface('my-model', 'path/to/model.')
    ```

    Args:
        repo_name (str): Name of the repository to create on Hugging Face Hub.
        model_path (str): Path to the local model file to upload. Can be safetensors or ckpt file.
        path_in_repo (str): Path to save the model in the repository.
    """

    hf_token = os.getenv('HF_Token')
    repo_id = 'formsKorea/' + repo_name
    try:
        api = HfApi()
        api.upload_file(path_or_fileobj=model_path, path_in_repo=path_in_repo, repo_id=repo_id, token=hf_token)
        logging.info(f"Model uploaded to Hugging Face Hub: huggingface.co/formsKorea/{repo_id}")
    except Exception as e:
        logging.error(f"Error uploading model to Hugging Face Hub: {e}")
        raise e


def upload_modelzoo(repo_name, model_path):
    """
    Uploads a local model to Hugging Face Hub. https://huggingface.co/formsKorea

    Example:
    ```
    upload_to_huggingface('my-model', 'path/to/model.')
    ```

    Args:
        repo_name (str): Name of the repository to create on Hugging Face Hub.
        model_path (str): Path to the local model file to upload. Can be safetensors or ckpt file.
    """

    hf_token = os.getenv('HF_Token')
    repo_id = 'formsKorea/' + repo_name
    try:
        create_repo(repo_id, private=True, exist_ok=False, token=hf_token)
        pipe = StableDiffusionPipeline.from_single_file(model_path)
        pipe.push_to_hub(repo_id, token=hf_token)
        logging.info(f"Model uploaded to Hugging Face Hub: huggingface.co/formsKorea/{repo_id}")
    except Exception as e:
        logging.error(f"Error uploading model to Hugging Face Hub: {e}")
        raise e


if __name__ == "__main__":
    upload_to_huggingface('majicmixrealistic-v7',
                          '/home/hamna/PycharmProjects/stable-diffusion/'
                          'stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_v7.safetensors')
