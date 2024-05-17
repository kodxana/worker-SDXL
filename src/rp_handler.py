'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        base_pipe = base_pipe.to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        refiner_pipe = refiner_pipe.to("cuda")
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)
            self.base = future_base.result()
            self.refiner = future_refiner.result()

    def get_scheduler(self, scheduler_name, scheduler_config):
        schedulers = {
            "PNDM": PNDMScheduler,
            "LMSDiscrete": LMSDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "EulerDiscrete": EulerDiscreteScheduler,
            "DPMSolverMultistep": DPMSolverMultistepScheduler
        }
        return schedulers.get(scheduler_name, DDIMScheduler).from_config(scheduler_config)

    def _save_and_upload_images(self, images, job_id):
        image_urls = []
        for i, img in enumerate(images):
            img_path = f"/tmp/{job_id}_{i}.png"
            img.save(img_path)
            image_url = rp_upload.upload_image(img_path)
            image_urls.append(image_url)
        return image_urls

    def generate_image(self, job):
        job_input = job['input']
        validated_input = validate(job_input, INPUT_SCHEMA)
        if not validated_input["valid"]:
            return {"error": validated_input["msg"]}

        job_input = validated_input["data"]
        generator = torch.manual_seed(job_input['seed']) if job_input['seed'] is not None else None

        starting_image = job_input.get('image_url')
        if starting_image:
            init_image = load_image(starting_image).convert("RGB").to("cuda").half()
            output = self.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=init_image,
                generator=generator
            ).images
        else:
            image = self.base(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                height=job_input['height'],
                width=job_input['width'],
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                denoising_end=job_input['high_noise_frac'],
                output_type="latent",
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images

            output = self.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=image,
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images

        image_urls = self._save_and_upload_images(output, job['id'])

        results = {
            "images": image_urls,
            "image_url": image_urls[0],
            "seed": job_input['seed']
        }

        if starting_image:
            results['refresh_worker'] = True

        return results

runpod.serverless.start({"handler": ModelHandler().generate_image})
