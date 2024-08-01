import wandb
wandb.login(key="69fea85817147f437d88815d733f3e2ed9799c12")
api = wandb.Api()

run = api.run("xiaoxubeii-ai/co2-emission-estimation/3hcmhzv8")
run.upload_file("/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/inversion/co2emiss-transformer-sch1-win12-patch16-07301929/w_best.keras",
                root="/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/experiments/inversion/")

# run.file("co2emiss-transformer-win12-patch16-07301929/w_best.keras").download(replace=True)
