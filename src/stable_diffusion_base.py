import torch
import logging
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    LCMScheduler, 
    AutoencoderTiny
)

from src.image import Image
from src.prompt import Prompt

NOT_IMPLEMENTED_MESSAGE = "This method should be implemented in subclasses."


class StableDiffusionBase:

    def __init__(self, model_id, tiny, lcm, width, height, seed, batch_count, batch_size, n_random_tokens, n_steps, guidance, torch_compile):
        self.logger = logging.getLogger("ArtSpew")
        self.model_id = model_id
        self.width = width
        self.height = height
        self.seed = seed
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.n_random_tokens = n_random_tokens
        self.n_steps = n_steps
        self.cfg_scale = guidance
        self.pipe = None

        self.configure_pipeline(lcm)
        if tiny:
            self.load_tiny_vae()
        if torch_compile:
            self.setup_torch_compilation()

    def load_pipeline(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def get_tiny_vae_model_id(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def get_filename_prefix(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def get_tokenizers(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def get_text_encoders(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def get_lcm_adapter_id(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def configure_pipeline(self, lcm):
        self.pipe = self.load_pipeline()
        self.setup_scheduler(self.pipe, lcm)
        self.configure_memory_format(self.pipe)

    def setup_scheduler(self, pipe, lcm):
        if lcm:
            self.logger.info("Using LCM.")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            self.load_and_fuse_lcm()
        else:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    @staticmethod
    def configure_memory_format(pipe):
        pipe.to("cuda")
        pipe.unet.to(memory_format=torch.channels_last)

    def setup_torch_compilation(self):
        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder, mode='max-autotune')
        self.pipe.unet = torch.compile(self.pipe.unet, mode='max-autotune')
        self.pipe.vae = torch.compile(self.pipe.vae, mode='max-autotune')
        self.perform_warmup()

    def perform_warmup(self):
        self.logger.info(
            "Starting warmup generation of two images. "
            "If using compile() and this is the first run it will add a number of minutes of extra time before it starts generating. "
            "Once the compile is done for a particular batch size there will only be something like a 35 seconds delay on a fast system each time you run after this. "
            "Obviously there is no reason to use compile unless you are going to generate hundreds of images."
        )
        with torch.inference_mode():
            for _ in [1, 2]:
                prompt = Prompt(self.get_tokenizers(), self.get_text_encoders(), "The cat in the hat is fat", self.n_random_tokens, self.batch_size)
                prompt.prepare()
                self.pipe(
                    prompt_embeds=prompt.embeds,
                    pooled_prompt_embeds=prompt.pooled_embeds,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=8,
                    guidance_scale=0,
                    lcm_origin_steps=50,
                    output_type="pil",
                    return_dict=False
                )

    def load_and_fuse_lcm(self):
        adapter_id = self.get_lcm_adapter_id()
        self.pipe.load_lora_weights(adapter_id)
        self.pipe.fuse_lora()

    def load_tiny_vae(self):
        vae_model_id = self.get_tiny_vae_model_id()
        self.pipe.vae = AutoencoderTiny.from_pretrained(vae_model_id, torch_device='cuda', torch_dtype=torch.float16)
        self.pipe.vae = self.pipe.vae.cuda()

    def generate_images(self, initial_text):
        processed_images = []
        for idx in range(self.batch_count):
            prompt = Prompt(
                self.get_tokenizers(),
                self.get_text_encoders(),
                self.pipe.unet,
                initial_text,
                self.n_random_tokens,
                self.batch_size
            )
            prompt.prepare()
            images = self.pipe(
                width=self.width,
                height=self.height,
                num_inference_steps=self.n_steps,
                prompt_embeds=prompt.embeds,
                pooled_prompt_embeds=prompt.pooled_embeds,
                guidance_scale=self.cfg_scale,
                lcm_origin_steps=50,
                output_type="pil",
                return_dict=False
            )[0]
            for image_idx, image in enumerate(images):
                settings = {
                    "steps": self.n_steps,
                    "sampler": "Euler a",
                    "cfg_scale": self.cfg_scale,
                    "seed": self.seed,
                    "width": self.width,
                    "height": self.height,
                    "model_id": self.model_id
                }
                processed_images.append(Image(image, prompt.text[image_idx], settings))

        return processed_images
