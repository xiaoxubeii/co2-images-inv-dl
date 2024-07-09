import kaggle
import os
import zipfile
import json
import subprocess


def upload_kaggle(username, api_key, dataset, model_path="w_last.keras"):
    with zipfile.ZipFile(model_path, "w") as zipf:
        zipf.write(model_path)

    init_kaggle(username, api_key)
    ds = kaggle.api.dataset_metadata(dataset, None)
    if ds and len(ds) > 0:
        ds = vars(ds[0])
        import pdb
        pdb.set_trace()


def init_kaggle(username, api_key):
    KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
    api_dict = {"username": username, "key": api_key}
    with open(f"{KAGGLE_CONFIG_DIR}/kaggle.json", "w", encoding='utf-8') as f:
        json.dump(api_dict, f)
    cmd = f"chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json"
    output = subprocess.check_output(cmd.split(" "))
    output = output.decode(encoding='UTF-8')
    print(output)


upload_kaggle("xiaoxubeii", "79b901941ec217aa258eaf8aec68e918",
              "inv_weights")
