import os
from huggingface_hub import snapshot_download

REPO_ID = "SyedHarshath/traffic-analytics-yolo"

def download_model():
    local_dir = "model_data"
    if not os.path.exists(os.path.join(local_dir,"saved_model.pb")):
        print("Downloading the SavedModel from Hugging Face")

        snapshot_download(repo_id=REPO_ID,local_dir=local_dir,local_dir_use_symlinks=False)

        print("Model Downloaded successfully!")

    return local_dir