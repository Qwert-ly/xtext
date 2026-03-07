import os
from huggingface_hub import hf_hub_download


# def txt_to_parquet(folder_path, output_name):
#     from datasets import Dataset
#     data = []
#     for file in os.listdir(folder_path):
#         if file.endswith(".txt"):
#             with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
#                 data.append({"title": file, "text": f.read()})
#
#     ds = Dataset.from_list(data)
#     ds.to_parquet(f"{output_name}.parquet")


# txt_to_parquet("./xtext/ctext - all - slice", "ctext - all - slice")

for f in ["ctext - all - slice.parquet", "ctext - 副本 - 副本.parquet", "ctext - 白话.parquet"]:
    hf_hub_download(
        repo_id='Nulll-Official/ctext',
        filename=f,
        repo_type='dataset',
        local_dir=os.path.dirname(os.path.abspath(__file__)),
        local_dir_use_symlinks=False
    )
