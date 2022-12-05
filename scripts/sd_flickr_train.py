import os 
import json
from PIL import Image
from typing import Optional
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing
import logging

from scripts.txt2img_call import txt2img


logger = logging.getLogger(__name__)


AUTH_TOKEN = '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'


def build_img2id(blip_flickr_train_file="/home/rdp455/lamp/data/blip_annotations/flickr30k_train.json"):
    img2id = {}
    with open(blip_flickr_train_file, "r") as input_json:
        train = json.load(input_json)
        for item in train:
            img2id[item["image"].replace("flickr30k-images/", "")]=item["image_id"]
    return img2id


def main(args):

    logger.info("load stable diffusion model...")

    stable_diffusion = txt2img(
        ckpt="sd-v1-4.ckpt",
        skip_grid=True,
        n_samples=5,
        plms=True,
        seed=42,
        scale=5,
        ddim_steps=50,
    )

    img2id = build_img2id() # convert to blip format would need this id

    # read image annotations
    logger.info("input file is :", args.annotation_file)
    image_annotations = []
    with open(args.annotation_file, "r") as input_json:
        for line in input_json.readlines():
            cur_img = json.loads(line)
            image_annotations.append(cur_img)

    # if args.topk is not None:
    #     image_annotations = image_annotations[:args.topk]

    # save image
    if not os.path.exists(os.path.join(args.output_folder, "images")):
        os.makedirs(os.path.join(args.output_folder, "images"))
        

    updated_annotation_file = os.path.basename(args.annotation_file).replace(".jsonl", "_generated.json")
    updated_annotations = []
    
    for i, anno in enumerate(tqdm(image_annotations)):
        sentence = anno["sentences"][args.anno_idx]
        img_path = os.path.join(args.image_dir, anno["img_path"])
        img_id = anno["id"]
        blip_image_id = img2id.get(img_path)
        
        # get generated images
        with autocast("cuda"):
            sd_imgs, time = stable_diffusion(prompt=str(sentence))
        gimg = sd_imgs[0]

        
        img_save_name = "%s_ann_%s.jpg" % (img_id, str(args.anno_idx))
        img_save_path = os.path.join(args.output_folder, "images", img_save_name)
        gimg.save(img_save_path)
        print("img saved to path: ", img_save_path)

        output_entry = {
            "image": img_save_path,
            "caption": sentence,
            "image_id": blip_image_id
            }

        updated_annotations.append(output_entry)
        # save to blip format
    
    logger.info("generated %d new entries for BLIP train..." % len(updated_annotations))

    with open(os.path.join(args.output_folder, updated_annotation_file), "w") as output_file:
        json.dump(updated_annotation_file, output_file)
    logger.info("annotations with generated images are updated to %s" % updated_annotation_file)


if __name__ == "__main__":
    multiprocessing.get_context('spawn')

    parser = ArgumentParser()
    parser.add_argument(
        "--anno_idx", dest="anno_idx", required=False,
        type=int, default=1, help="ith caption in the annotation list")
    parser.add_argument(
        "--image_dir", dest="image_dir",
        type=str, default="/image/nlp-datasets/emanuele/data/flickr30k/images",
        help="image directory that saves input images")
    parser.add_argument(
        "--annotation_file", dest="annotation_file",
        type=str, default="/image/nlp-datasets/emanuele/data/flickr30k/annotations/train_ann.jsonl",
        help="path to the annotation file")
    parser.add_argument(
        "--output_folder", dest="output_folder",
        default="/home/rdp455/lamp/data/sd/",
        type=str, help="output folder to save generated images"
    )

    args = parser.parse_args()
    main(args)












