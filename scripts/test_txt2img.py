import os 
import json
from PIL import Image

from scripts.txt2img_call import txt2img

import wandb
wandb.init(project="sd-synth")

with open("/home/rff/Documents/mountPoint/dataset_coco.json", "r") as fp:
    mscoco = json.load(fp)
    
stable_diffusion = txt2img(
    ckpt="sd-v1-4.ckpt",
    skip_grid=True,
    n_samples=5,
    plms=True,
    seed=42,
    scale=5,
    ddim_steps=50,
)

table = wandb.Table(columns=["Org", "SD0", "Prompt", "Time [sec]"])
base = "/home/rff/Documents/mountPoint/images"
# outdir = "/home/rff/Documents/mountPoint"
for idx in range(12,14):
    filepath = mscoco['images'][idx]['filepath']
    filename = mscoco['images'][idx]['filename']
    sentence = mscoco['images'][idx]['sentences'][0]['raw']
    # root, ext = os.path.splitext(filename)
    sd_imgs, time = stable_diffusion(prompt=str(sentence))
    sd_imgs = sd_imgs[0]
    img = Image.open(os.path.join(base, filepath, filename))
    table.add_data(
        wandb.Image(img, caption="Original"),
        wandb.Image(sd_imgs, caption="SD"),
        sentence,
        time
        )
wandb.log({"SD": table})
    # data = [img, ]
    # print(len(sd_imgs), time)
    
