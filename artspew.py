import time
import random
import os
import argparse
import torch
import numpy as np
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler, LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline


class StableDiffusionBase:
    def __init__(self, model_id, tiny, lcm, width, height, batch_size, n_tokens, n_steps, guidance, torch_compile):

        self.model_id = model_id
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.n_steps = n_steps
        self.guidance = guidance

        pipe = self.load_pipeline()

        if lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            # ETA to first complaint I don't have your fav Scheduler: 4.07us
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.to("cuda")
        pipe.unet.to(memory_format=torch.channels_last)

        if tiny:
            print("Using Tiny VAE")
            self.load_tiny_vae()

        if lcm:
            print("Using LCM")
            self.load_and_fuse_lcm()

        self.pipe = pipe

        last_sequence_number = -1

        # Use os.scandir() to efficiently list and filter files
        files = [entry.name for entry in os.scandir('spew')
                if entry.name.startswith(self.get_filename_prefix())]

        if len(files) > 0:
            sortedFiles = sorted(files, key=lambda x: int(x.split('-')[1]))
            last_sequence_number = int(sortedFiles[-1].split('-')[1])
        else:
            last_sequence_number = -1
        self.last_sequence_number = last_sequence_number

        if torch_compile:
            pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
            pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
            pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

            # Warmup
            print('\nStarting warmup generation of two images.')
            print('If using compile() and this is the first run it will add a number of minutes of extra time before it starts generating.  Once the compile is done for a paticular batch size there will only be something like a 35 seconds delay on a fast system each time you run after this')
            print('Obviously there is no reason to use compile unless you are going to generate hundreds of images')
            with torch.inference_mode():
                for _ in [1,2]:
                    pe, ppe = self.dwencode('The cat in the hat is fat', self.batch_size, 5)
                    pipe(
                        prompt_embeds = pe,
                        pooled_prompt_embeds = ppe,
                        width=self.width, height=self.height,
                        num_inference_steps=8,
                        guidance_scale=0,
                        lcm_origin_steps=50,
                        output_type="pil", return_dict=False)

    def load_pipeline(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_tiny_vae(self):
        from diffusers import AutoencoderTiny
        vae_model_id = self.get_tiny_vae_model_id()
        self.pipe.vae = AutoencoderTiny.from_pretrained(vae_model_id, torch_device='cuda', torch_dtype=torch.float16)
        self.pipe.vae = self.pipe.vae.cuda()

    def get_tiny_vae_model_id(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_filename_prefix(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_tokenizers(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_text_encoders(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_and_fuse_lcm(self):
        adapter_id = self.get_lcm_adapter_id()
        self.pipe.load_lora_weights(adapter_id)
        self.pipe.fuse_lora()

    def get_lcm_adapter_id(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def dwencode(self, prompt, batch_size, n_tokens):
        pipe = self.pipe

        if prompt == None:
            prompt = ''

        if n_tokens < 0 or n_tokens > 75:
            raise BaseException("n random tokens must be between 0 and 75")

        prompt = batch_size * [ prompt ]

        tokenizers = self.get_tokenizers()
        text_encoders = self.get_text_encoders()

        if n_tokens > 0:
            randIIs = torch.zeros((batch_size, n_tokens), dtype=torch.int32)
            for bIdx in range(batch_size):
                randIIs[bIdx] = torch.randint(low=0, high=49405, size=(n_tokens,))

        # Each prompt is a batch size of the prompts.  However,
        # in the below "prompts" are to deal with the two
        # tokenizers and encoders.  Confused?
        #prompts = [prompt, ["Women on spaceship", "Gorilla on building", "Children swimming", "Chicken riding a donkey"]]
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding = "max_length",
                max_length = tokenizer.model_max_length,
                truncation = True,
                return_tensors="pt",
            )

            # pl is prompt length in terms of user tokens
            # Find the end marker
            pl = np.where(text_inputs.input_ids[0] == 49407)[0][0] - 1
            if pl + n_tokens > 75:
                raise BaseException("Number of user prompt tokens and random tokens must be <= 75")

            text_input_ids = text_inputs.input_ids

            if n_tokens > 0:
                tii = text_inputs.input_ids
                # This appends nToken random tokens to the user prompt
                for i in range(batch_size):
                    tii[i][1+pl:1+pl+n_tokens] = randIIs[i]
                    tii[i][1+pl+n_tokens] = 49407
                text_input_ids = tii

            prompt_embeds = text_encoder(text_input_ids.to('cuda'), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            clip_skip = None
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        seqno = self.last_sequence_number + 1
        for bi in range(batch_size):
            print(f"{seqno:09d}-{bi:02d}: ", end='')
            for tid in text_input_ids[bi][1:1+pl+n_tokens]:
                print(f"{tokenizer.decode(tid)} ", end='')
            print('')

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device='cuda')

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
        if len(tokenizers) > 1:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
        else:
            pooled_prompt_embeds = None

        return prompt_embeds, pooled_prompt_embeds

    def generate_images(self, prompt):
        # Common image generation code
        pe, ppe = self.dwencode(prompt, self.batch_size, self.n_tokens)
        images = self.pipe(
            width=self.width, height=self.height,
            num_inference_steps=self.n_steps,
            prompt_embeds=pe,
            pooled_prompt_embeds = ppe,
            guidance_scale=self.guidance,
            output_type="pil", return_dict=False
        )[0]
        self.last_sequence_number += 1
        btchidx = 0
        for img in images:
            img.save("spew/" + self.get_filename_prefix() + f"{self.last_sequence_number:09d}-{btchidx:02d}.jpg")
            btchidx += 1


class StableDiffusionSD15(StableDiffusionBase):
    def load_pipeline(self):
        if self.model_id.endswith('.safetensors') or self.model_id.endswith('.ckpt'):
            pipe = StableDiffusionPipeline.from_single_file(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False)
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


class StableDiffusionSDXL(StableDiffusionBase):
    def load_pipeline(self):
        if self.model_id.endswith('.safetensors') or self.model_id.endswith('.ckpt'):
            pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--xl', action=argparse.BooleanOptionalAction,
                        help='Use SDXL')
    parser.add_argument('-m', '--model_id', type=str, default='auto',
                        help='Specify the input file')
    parser.add_argument('-p', '--prompt', type=str,
                        help='Specify the start of the prompt')
    parser.add_argument('--width', type=int, default=-1,
                        help='Image width, -1 for auto')
    parser.add_argument('--height', type=int, default=-1,
                        help='Image height, -1 for auto')
    parser.add_argument('-c', '--batch-count', type=int, default=1,
                        help='Number of batches to do')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('-s', '--nSteps', type=int, default=-1,
                        help='Number of inference steps, -1 for auto')
    parser.add_argument('-n', '--nRandTokens', type=int, default=0,
                        help='Number of random tokens added')
    parser.add_argument('-l', '--lcm', action='store_true',
                        help='Use LCM')
    parser.add_argument('-t', '--tiny-vae', action='store_true',
                        help='Use the tiny VAE')
    parser.add_argument('-g', '--guidance', type=float, default=-1,
                        help='Guidance value, -1 for auto')
    parser.add_argument('-z', '--torch-compile', action='store_true',
                        help='Using torch.compile for faster inference')
    
    args = parser.parse_args()
 
    if args.model_id == 'auto':
        if args.xl:
            args.model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        else:
            args.model_id = 'runwayml/stable-diffusion-v1-5'

    if args.width == -1:
        if args.xl:
            args.width = 1024
        else:
            args.width = 512

    if args.height == -1:
        if args.xl:
            args.height = 1024
        else:
            args.height = 512

    if args.nSteps == -1:
        if args.lcm:
            args.nSteps = 8.
        else:
            args.nSteps = 20

    if args.guidance == -1:
        if args.lcm:
            args.guidance = 0.
        else:
            args.guidance = 8.

    return args


def main():
    args = parse_arguments()

    if not os.path.exists('spew'):
        os.makedirs('spew')

    print(f"\ngenerating {args.batch_count*args.batch_size} images with {args.nSteps} LCM steps")
    print(f"It can take a few minutes to download the model the very first run.")
    print("After that it can take 10s of seconds to load the rather large sdxl model.")

    seed = random.randint(0, 2147483647)
    torch.manual_seed(seed)

    if args.xl:
        model = StableDiffusionSDXL(args.model_id, args.tiny_vae, args.lcm, args.width, args.height, args.batch_size, args.nRandTokens, args.nSteps, args.guidance, args.torch_compile)
    else:
        model = StableDiffusionSD15(args.model_id, args.tiny_vae, args.lcm, args.width, args.height, args.batch_size, args.nRandTokens, args.nSteps, args.guidance, args.torch_compile)

    model.generate_images(args.prompt)


if __name__ == "__main__":
    main()
