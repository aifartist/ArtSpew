from src.prompt_sdxl import PromptSDXL
from src.stable_diffusion_base import StableDiffusionBase
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
import torch


class StableDiffusionSDXL(StableDiffusionBase):
    def load_pipeline(self):
        # Is it a local file?
        if self._model_id.endswith('.safetensors') or self._model_id.endswith('.ckpt'):
            pipe = StableDiffusionXLPipeline.from_single_file(
                self._model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False)
        return pipe

    def get_tiny_vae_model_id(self):
        return 'madebyollin/taesdxl'

    def get_lcm_adapter_id(self):
        return 'latent-consistency/lcm-lora-sdxl'

    def get_filename_prefix(self):
        return "sdxl-"

    def get_tokenizers(self):
        return [self.pipe.tokenizer, self.pipe.tokenizer_2]
    
    def get_text_encoders(self):
        return [self.pipe.text_encoder, self.pipe.text_encoder_2]

    @staticmethod
    def get_prompt_class():
        return PromptSDXL
