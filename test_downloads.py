from omegaconf import OmegaConf
import wandb
import os

default_config = {
    "working_dir": "/kaggle/working",
    "data.path.directory": "/kaggle/input/inv-pp/data_paper_inv_pp",
    # "source_code": source_code,
    "apikey": "69fea85817147f437d88815d733f3e2ed9799c12",
}


def default():
    return default_config


def update(cfg, u):
    for k, v in u.items():
        OmegaConf.update(cfg, k, v, merge=True)


def download_tests(last_step_cfg, cfg):
    dest_dir = os.path.join(cfg.working_dir, "res", cfg.model.type)
    wandb.login(key=cfg.apikey)
    api = wandb.Api()
    tests = {}
    for r in cfg.tests:
        run = api.run(r["run_path"])
        run.file(os.path.join(run.name, "metric.json")).download(
            root=os.path.join(dest_dir, run.name))
        tests["run.exp_name"] = {
            "metric_path": os.path.join(dest_dir, run.name, "metric.json")
        }

    return OmegaConf.create({
        "tests": tests
    })


runs = ["xiaoxubeii-ai/co2-emission-estimation/cvqzv9bs",
        "xiaoxubeii-ai/co2-emission-estimation/8b86s2h0", "xiaoxubeii-ai/co2-emission-estimation/clna27br"]
cfg = OmegaConf.create()
update(cfg, default())

tests = []
for r in runs:
    tests.append({
        "run_path": r,

    })

update(cfg, {
    "tests": tests,
    "model.type": "inversion"
})

download_tests(None, cfg)


#     for k, v in cfg.tests.items():
#         with open(v["metric_path"], 'r') as f:
#             metric = json.load(f)
#             metric["method"] = v["model_name"]
#             metrics.append(metric)
