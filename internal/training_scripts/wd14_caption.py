import gradio as gr
from easygui import msgbox
import subprocess
import os
from typing import List
import json
from internal.training_scripts.library.custom_logging import setup_logging
from internal.training_scripts.library.class_command_executor import CommandExecutor

# Set up logging
log = setup_logging()
# def load_json(json_filepath):
#     """
#       Load_json file
#
#       Args:
#           load_json(json_filepath): load json file
#
#       Returns:
#           json
#       """
#     with open(json_filepath, 'r') as file:
#         return json.load(file)

def caption_images(
    train_data_dir : str,
    prefix_tags,
    caption_extension : str =".txt",
    batch_size : int = 8,
    general_threshold : float =  0.35,
    character_threshold : float = 0.35,
    replace_underscores : bool = True,
    model : str = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    recursive : bool = False,
    max_data_loader_n_workers : int = 2,
    debug : bool = False,
    undesired_tags : str = "",
    frequency_tags : bool = True,
    onnx : bool = False,
    append_tags : bool = False,
    force_download : bool = False
):
    # Check for images_dir_input
    if train_data_dir == '':
        msgbox('Image folder is missing...')
        return

    if caption_extension == '':
        msgbox('Please provide an extension for the caption files.')
        return

    log.info(f'Captioning files in {train_data_dir}...')
    run_cmd = f'accelerate launch "./internal/training_scripts/tag_images_by_wd14_tagger.py"'
    run_cmd += f' --batch_size={int(batch_size)}'
    run_cmd += f' --general_threshold={general_threshold}'
    run_cmd += f' --character_threshold={character_threshold}'
    run_cmd += f' --caption_extension="{caption_extension}"'
    run_cmd += f' --model="{model}"'
    run_cmd += (
        f' --max_data_loader_n_workers="{int(max_data_loader_n_workers)}"'
    )

    if recursive:
        run_cmd += f' --recursive'
    if debug:
        run_cmd += f' --debug'
    if replace_underscores:
        run_cmd += f' --remove_underscore'
    if frequency_tags:
        run_cmd += f' --frequency_tags'
    if onnx:
        run_cmd += f' --onnx'
    if append_tags:
        run_cmd += f' --append_tags'
    if force_download:
        run_cmd += f' --force_download'

    if not undesired_tags == '':
        run_cmd += f' --undesired_tags="{undesired_tags}"'

    run_cmd += f' "{train_data_dir}"'
    run_cmd += f' "--prefix_tags={prefix_tags}"'
    log.info(run_cmd)

    # Run the command
    # print(os.name)
    # if os.name == 'posix':
    #     os.system(run_cmd)
    # else:
    #     subprocess.run(run_cmd)
    #
    # log.info('...captioning done')

    return run_cmd

if __name__ == "__main__":
    # config = load_json("/home/hamna/kyle/Project/VisionForge/internal/training_scripts/example_cap_json/man_caption.json")
    caption_images(train_data_dir='/home/hamna/kyle/Project/aaa/20240502', prefix_tags=['dels','dels asian man'])
