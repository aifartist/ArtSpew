import torch
import logging
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    LCMScheduler, 
    AutoencoderTiny
)

from src.image import Image

NOT_IMPLEMENTED_MESSAGE = "This method should be implemented in subclasses."


class StableDiffusionBase:

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipe = None

        self.model_id = kwargs.pop('model_id')
        self.tiny_vae = kwargs.pop('tiny_vae')
        self.lcm = kwargs.pop('lcm')
        self.width = kwargs.pop('width')
        self.height = kwargs.pop('height')
        self.seed = kwargs.pop('seed')
        self.batch_count = kwargs.pop('batch_count')
        self.batch_size = kwargs.pop('batch_size')
        self.n_random_tokens = kwargs.pop('n_random_tokens')
        self.n_steps = kwargs.pop('n_steps')
        self.guidance_scale = kwargs.pop('guidance_scale')
        self.torch_compile = kwargs.pop('torch_compile')

        if len(kwargs) > 0:
            raise ValueError(f"Unknown arguments: {kwargs}")

        self.configure_pipeline(self.lcm)
        if self.tiny_vae:
            self.load_tiny_vae()
        if self.torch_compile:
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

    @staticmethod
    def get_prompt_class():
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
                prompt_class = self.get_prompt_class()
                prompt = prompt_class(self.get_tokenizers(), self.get_text_encoders(), "The cat in the hat is fat", self.n_random_tokens, self.batch_size)
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
            prompt_class = self.get_prompt_class()
            prompt = prompt_class(
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
                guidance_scale=self.guidance_scale,
                lcm_origin_steps=50,
                output_type="pil",
                return_dict=False
            )[0]
            for image_idx, image in enumerate(images):
                settings = {
                    "steps": self.n_steps,
                    "sampler": "Euler a",
                    "cfg_scale": self.guidance_scale,
                    "seed": self.seed,
                    "width": self.width,
                    "height": self.height,
                    "model_id": self.model_id
                }
                processed_images.append(Image(image, prompt.text[image_idx], settings))

        return processed_images
