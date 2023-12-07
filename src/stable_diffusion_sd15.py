from src.prompt_sd15 import PromptSD15
from src.stable_diffusion_base import StableDiffusionBase
from diffusers import StableDiffusionPipeline
import torch


class StableDiffusionSD15(StableDiffusionBase):
    def load_pipeline(self):
        if self._model_id.endswith('.safetensors') or self._model_id.endswith('.ckpt'):
            pipe = StableDiffusionPipeline.from_single_file(
                self._model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False
            )
        return pipe

    def get_tiny_vae_model_id(self):
        return 'madebyollin/taesd'

    def get_lcm_adapter_id(self):
        return "latent-consistency/lcm-lora-sdv1-5"
    
    def get_filename_prefix(self):
        return "sd15-"
    
    def get_tokenizers(self):
        return [self.pipe.tokenizer]
    
    def get_text_encoders(self):
        return [self.pipe.text_encoder]

    @staticmethod
    def get_prompt_class():
        return PromptSD15
