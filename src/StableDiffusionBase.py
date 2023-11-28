import torch
import numpy as np
import logging
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    LCMScheduler, 
    AutoencoderTiny
)

NOT_IMPLEMENTED_MESSAGE = "This method should be implemented in subclasses."


class StableDiffusionBase:

    def __init__(self, model_id, tiny, lcm, width, height, batch_count, batch_size, n_random_tokens, n_steps, guidance, torch_compile):
        self.logger = logging.getLogger("ArtSpew")
        self.model_id = model_id
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.n_random_tokens = n_random_tokens
        self.n_steps = n_steps
        self.guidance = guidance
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
        raise NotImplementedError("This method should be implemented in subclasses.")

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
                prompt_embeds, pooled_prompt_embeds, _ = self.dwencode('The cat in the hat is fat', self.batch_size, 5)
                self.pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
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

    def dwencode(self, prompt, batch_size, n_random_tokens):
        if prompt is None:
            prompt = ''

        if not 0 <= n_random_tokens <= 75:
            raise ValueError("n random tokens must be between 0 and 75")

        prompt = [prompt] * batch_size  # Replicate the prompt for the batch
        tokenizers = self.get_tokenizers()
        text_encoders = self.get_text_encoders()

        # Generate random tokens if needed
        random_tokens = self.generate_random_tokens(batch_size, n_random_tokens) if n_random_tokens > 0 else None

        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        text_inputs_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = self.tokenize_prompt(tokenizer, prompt)
            text_inputs_list.append(text_inputs)
            prompt_length = self.find_prompt_length(text_inputs)

            # Append random tokens to the user prompt if needed
            if n_random_tokens > 0:
                text_inputs.input_ids = self.append_random_tokens(text_inputs.input_ids, random_tokens, prompt_length, n_random_tokens)

            prompt_embeds, pooled_prompt_embeds = self.encode_text(text_encoder, text_inputs.input_ids)
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        decoded_prompts = []
        for encoded_prompt in text_inputs_list[0].input_ids:
            decoded_prompt = tokenizers[0].decode(encoded_prompt, skip_special_tokens=True)
            decoded_prompts.append(decoded_prompt)
            self.logger.info("Prompt: " + decoded_prompt)

        # Concatenate and prepare embeddings for the model
        return self.prepare_embeddings_for_model(prompt_embeds_list, pooled_prompt_embeds_list, decoded_prompts)

    def generate_random_token(self):
        max_token_id = self.pipe.tokenizer.vocab_size
        special_token_ids = set(self.pipe.tokenizer.all_special_ids)

        while True:
            token_id = torch.randint(low=0, high=max_token_id, size=(1,), dtype=torch.int32).item()
            if token_id not in special_token_ids:
                return token_id
            
    def generate_random_tokens(self, batch_size, n_random_tokens):
        random_tokens = torch.zeros((batch_size, n_random_tokens), dtype=torch.int32)
        for i in range(batch_size):
            for j in range(n_random_tokens):
                random_tokens[i, j] = self.generate_random_token()
        return random_tokens

    @staticmethod
    def tokenize_prompt(tokenizer, prompt):
        return tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def find_prompt_length(self, text_inputs):
        return np.where(text_inputs.input_ids[0] == self.pipe.tokenizer.eos_token_id)[0][0] - 1

    def append_random_tokens(self, input_ids, random_tokens, prompt_length, n_random_tokens):
        if prompt_length + n_random_tokens > 75:
            raise ValueError("Number of user prompt tokens and random tokens must be <= 75")
        for i in range(len(input_ids)):
            input_ids[i][1+prompt_length:1+prompt_length+n_random_tokens] = random_tokens[i]
            input_ids[i][1+prompt_length+n_random_tokens] = self.pipe.tokenizer.eos_token_id
        return input_ids

    @staticmethod
    def encode_text(text_encoder, text_input_ids):
        prompt_embeds = text_encoder(text_input_ids.to('cuda'), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]  # Use the penultimate layer
        return prompt_embeds, pooled_prompt_embeds

    def prepare_embeddings_for_model(self, prompt_embeds_list, pooled_prompt_embeds, decoded_prompts):
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=self.pipe.unet.dtype, device='cuda')
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1).view(bs_embed, seq_len, -1)

        if len(prompt_embeds_list) > 1:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed, -1)
        else:
            pooled_prompt_embeds = None

        return prompt_embeds, pooled_prompt_embeds, decoded_prompts
    
    def generate_images(self, prompt):
        processed_images = []
        for idx in range(self.batch_count):
            prompt_embeds, pooled_prompt_embeds, decoded_prompts = self.dwencode(prompt, self.batch_size, self.n_random_tokens)
            images = self.pipe(
                width=self.width,
                height=self.height,
                num_inference_steps=self.n_steps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=self.guidance,
                lcm_origin_steps=50,
                output_type="pil",
                return_dict=False
            )[0]
            for image_idx, image in enumerate(images):
                processed_images.append({"image": image, "prompt": decoded_prompts[image_idx]})

        return processed_images
