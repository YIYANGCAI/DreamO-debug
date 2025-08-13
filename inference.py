import os
os.environ['HF_HOME'] = '/aifs4su/caiyiyang/cache'
import argparse

import cv2
import gradio as gr
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from optimum.quanto import freeze, qint8, quantize
from PIL import Image
from torchvision.transforms.functional import normalize

from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import (
    img2tensor,
    resize_numpy_image_area,
    resize_numpy_image_long,
    tensor2img,
)
from tools import BEN2

def get_device():
    """Automatically detect the best available device"""
    device = 'auto'  # Default to auto, can be overridden by command line argument or environment variable
    if device != 'auto':
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class Generator:
    def __init__(self):
        offload = False
        int8 = False
        no_turbo = False
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # preprocessing models
        # background remove model: BEN2
        self.bg_rm_model = BEN2.BEN_Base().to(self.device).eval()
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        # face crop and align tool: facexlib
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        if offload:
            self.ben_to_device(torch.device('cpu'))
            self.facexlib_to_device(torch.device('cpu'))

        # load dreamo
        model_root = 'black-forest-labs/FLUX.1-dev'
        dreamo_pipeline = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        dreamo_pipeline.load_dreamo_model(self.device, use_turbo=not no_turbo)
        if int8:
            print('start quantize')
            quantize(dreamo_pipeline.transformer, qint8)
            freeze(dreamo_pipeline.transformer)
            quantize(dreamo_pipeline.text_encoder_2, qint8)
            freeze(dreamo_pipeline.text_encoder_2)
            print('done quantize')
        self.dreamo_pipeline = dreamo_pipeline.to(self.device)
        if offload:
            self.dreamo_pipeline.enable_model_cpu_offload()
            self.dreamo_pipeline.offload = True
        else:
            self.dreamo_pipeline.offload = False

    def ben_to_device(self, device):
        self.bg_rm_model.to(device)

    def facexlib_to_device(self, device):
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)

    @torch.no_grad()
    def get_align_face(self, img):
        # the face preprocessing code is same as PuLID
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            return None
        align_face = self.face_helper.cropped_faces[0]

        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, input)
        face_features_image = tensor2img(face_features_image, rgb2bgr=False)

        return face_features_image

@torch.inference_mode()
def generate_image(
    generator,
    ref_image1,
    ref_image2,
    ref_task1,
    ref_task2,
    prompt,
    width,
    height,
    ref_res,
    num_steps,
    guidance,
    seed,
    true_cfg,
    cfg_start_step,
    cfg_end_step,
    neg_prompt,
    neg_guidance,
    first_step_guidance,
):
    print(prompt)
    offload = False
    ref_conds = []
    debug_images = []

    ref_images = [ref_image1, ref_image2]
    print(ref_images)
    ref_tasks = [ref_task1, ref_task2]

    for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
        if ref_image is not None:
            if ref_task == "id":
                if offload:
                    generator.facexlib_to_device(generator.device)
                ref_image = resize_numpy_image_long(ref_image, 1024)
                ref_image = generator.get_align_face(ref_image)
                if offload:
                    generator.facexlib_to_device(torch.device('cpu'))
            elif ref_task != "style":
                if offload:
                    generator.ben_to_device(generator.device)
                ref_image = generator.bg_rm_model.inference(Image.fromarray(ref_image))
                if offload:
                    generator.ben_to_device(torch.device('cpu'))
            if ref_task != "id":
                ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
            debug_images.append(ref_image)
            ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    'img': ref_image,
                    'task': ref_task,
                    'idx': idx + 1,
                }
            )

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()

    image = generator.dreamo_pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        ref_conds=ref_conds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=true_cfg,
        true_cfg_start_step=cfg_start_step,
        true_cfg_end_step=cfg_end_step,
        negative_prompt=neg_prompt,
        neg_guidance_scale=neg_guidance,
        first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
    ).images[0]

    return image

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref1', type=str)
    parser.add_argument('--ref2', type=str)
    parser.add_argument('--task1', type=str, default='id', choices=['id', 'ip', 'bg'], help='task for the first reference image')
    parser.add_argument('--task2', type=str, default='id', choices=['id', 'ip', 'bg'], help='task for the second reference image')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--output', type=str, default='output.png', help='output image file path')
    args = parser.parse_args()
    # generator
    generator = Generator()
    # default parameters
    ref_image1 = cv2.imread(args.ref1)
    ref_image1 = cv2.cvtColor(ref_image1, cv2.COLOR_BGR2RGB)
    print(ref_image1)
    ref_image2 = cv2.imread(args.ref2)
    ref_image2 = cv2.cvtColor(ref_image2, cv2.COLOR_BGR2RGB)
    print(ref_image2)
    prompt = args.prompt
    ref_task1 = args.task1
    ref_task2 = args.task2
    width  = 1024
    height = 1024
    ref_res = 512
    num_steps = 50
    guidance = 7.5
    seed = 42
    true_cfg = 1.0
    cfg_start_step = 0
    cfg_end_step = 0
    neg_prompt = "Low quality, blurry, deformed, distorted, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality"
    neg_guidance = 3.5
    first_step_guidance = 0

    image = generate_image(
        generator,  
        ref_image1,
        ref_image2,
        ref_task1,
        ref_task2,
        prompt,
        width,
        height,
        ref_res,
        num_steps,
        guidance,
        seed,
        true_cfg,
        cfg_start_step,
        cfg_end_step,
        neg_prompt,
        neg_guidance,
        first_step_guidance
    )
    image.save(args.output)
    return image

main()