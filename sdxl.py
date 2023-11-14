import time
import random
import os, signal
import subprocess
from subprocess import check_output

import torch
import numpy as np

#from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import LCMScheduler

doCompile = False
lastSeqno = -1

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.benchmark_limit = 0

def dwencode(pipe, prompt: str, batch_size: int, nTokens: int):
    if nTokens < 0 or nTokens > 75:
        raise BaseException("n random tokens must be between 0 and 75")

    prompt = batch_size * [ prompt ]

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    if nTokens > 0:
        randIIs = torch.zeros((batch_size, nTokens), dtype=torch.int32)
        for bIdx in range(batch_size):
            randIIs[bIdx] = torch.randint(low=0, high=49405, size=(nTokens,))

    # Each prompt is a batch size of the prompts.  However,
    # in the below "prompts" are to deal with the two
    # tokenizers and encoders.  Confused?
    #prompts = [prompt, ["Women on spaceship", "Gorilla on building", "Children swimming", "Chicken riding a donkey"]]
    prompts = [prompt, prompt]
    prompt_embeds_list = []
    for idx, prompt, tokenizer, text_encoder in zip([0, 1], prompts, tokenizers, text_encoders):
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

        if pl + nTokens > 75:
            raise BaseException("Number of user prompt tokens and random tokens must be <= 75")

        text_input_ids = text_inputs.input_ids

        if nTokens > 0:
            tii = text_inputs.input_ids
            # This appends nToken random tokens to the user prompt
            for i in range(batch_size):
                tii[i][1+pl:1+pl+nTokens] = randIIs[i]
                tii[i][1+pl+nTokens] = 49407
                #print(f"{idx}-{i}: tii {tii[i][0:1+pl+nTokens+1]}")
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

    seqno = lastSeqno + 1
    for bi in range(batch_size):
        print(f"{seqno:09d}-{bi:02d}: ", end='')
        for tid in text_input_ids[bi][1:1+pl+nTokens]:
            print(f"{tokenizer.decode(tid)} ", end='')
        print('')

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device='cuda')

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

    # text_encoder 1
    # DW: len(prompt_enbeds.hidden_states) = 13
    # DW: len(prompt_enbeds.hidden_states[-2].shape = torch.Size([3, 77, 768])

    return prompt_embeds, pooled_prompt_embeds

def warmup(pipe, prompt, guidance):
    # Warmup
    print('\nStarting warmup generation of two images.')
    print('If using compile() and this is the first run it will add a number of minutes of extra time before it start generating.  Once the compile is done for a paticular batch size there will only be something like a 35 seconds delay on a fast system each time you run')
    print('    this can take an extra 35 seconds each time.')
    with torch.inference_mode():
        pe, ppe = dwencode(pipe, prompt, batchSize, nSteps)
        img = pipe(
            prompt = prompt,
            width=1024, height=1024,
            prompt_embeds = pe,
            pooled_prompt_embeds = ppe,
            num_inference_steps=8,
            guidance_scale=guidance,
            output_type="pil", return_dict=False)
        pe, ppe = dwencode(pipe, prompt, batchSize, nSteps)
        img = pipe(
            width=1024, height=1024,
            prompt_embeds = pe,
            pooled_prompt_embeds = ppe,
            num_inference_steps=8,
            guidance_scale=guidance,
            output_type="pil", return_dict=False)

def genit(pipe, guidance, prompt, nBatches, batchSize, nTokens, nSteps):
    global lastSeqno

    with torch.inference_mode():
        tm0 = time.time()
        for idx in range(0, nBatches):
            import random

            seed = random.randint(0, 2147483647)
            torch.manual_seed(seed)

            pe, ppe = dwencode(pipe, prompt, batchSize, nTokens)

            tm00 = time.time()
            images = pipe(
                width=1024, height=1024,
                num_inference_steps = nSteps,
                prompt_embeds = pe,
                pooled_prompt_embeds = ppe,
                guidance_scale=guidance,
                output_type="pil", return_dict=False)[0]
            print(f"time = {(1000*(time.time() - tm00)):5.2f} milliseconds")
            lastSeqno += 1
            btchidx = 0
            for img in images:
                img.save(f"spew/sdxl-{lastSeqno:09d}-{btchidx:02d}.jpg")
                btchidx += 1
        print(f"time = {time.time() - tm0}")

import sys
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define an option with a value
    parser.add_argument('-m', '--model_id', type=str, default='stabilityai/stable-diffusion-xl-base-1.0',
                        help='Specify the input file')
    parser.add_argument('-p', '--prompt', type=str,
                        help='Specify the input file')
    parser.add_argument('-c', '--batch-count', type=int, default=1,
                        help='Number of batches to do')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch Size')
    # Using default=-1 allows me to use 8 for LCM and 20 otherwise
    parser.add_argument('-s', '--nSteps', type=int, default=-1,
                        help='Number of inference steps')
    parser.add_argument('-n', '--nRandTokens', type=int, default=0,
                        help='Number of random tokens added')
    parser.add_argument('-l', '--lcm', action='store_true',
                        help='Use LCM')
    parser.add_argument('-t', '--tiny-vae', action='store_true',
                        help='Use the tiny VAE')
    parser.add_argument('-g', '--guidance', type=float, default=-1.,
                        help='Guidance value')

    args = parser.parse_args()

    if args.lcm:
       if args.nSteps == -1:
           args.nSteps = 8
       # This would be for SDXL LCM
       if args.guidance == -1:
           args.guidance = 0.
    else:
       args.guidance = 8.
       if args.nSteps == -1:
           args.nSteps = 20

    prompt = args.prompt
    guidance = args.guidance
    nBatches = args.batch_count
    batchSize = args.batch_size
    nTokens = args.nRandTokens
    nSteps = args.nSteps
    lcm = args.lcm
    tiny = args.tiny_vae

    if args.model_id.endswith('.safetensors') or args.model_id.endswith('.ckpt'):
        pipe = AutoPipelineForText2Image.from_single_file(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False)
    if lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        # ETA to first complaint I don't have your fav Scheduler: 4.07us
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.to("cuda")
    pipe.unet.to(memory_format=torch.channels_last)

    if tiny:
        from diffusers import AutoencoderTiny
        pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
        pipe.vae = pipe.vae.cuda()

    if lcm:
        # load and fuse lcm lora
        adapter_id = 'latent-consistency/lcm-lora-sdxl'
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()

    if doCompile:
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
        #pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')
        pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
        pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

        warmup(pipe, 'The cat in the hat is fat', guidance)

    print(f"\nprompt = {prompt}")
    print(f"guidance = {guidance}")
    print(f"Number of Batches = {nBatches}")
    print(f"Batch Size = {batchSize}")
    print(f"Number of added random tokens = {nTokens}")
    print(f"Number of inference steps = {nSteps}")

    if not os.path.exists('spew'):
        os.makedirs('spew')

    fNamePrefix = f"sdxl-"
    # Use os.scandir() to efficiently list and filter files
    files = [entry.name for entry in os.scandir('spew')
             if entry.name.startswith(fNamePrefix)]

    if len(files) > 0:
        sortedFiles = sorted(files, key=lambda x: int(x.split('-')[1]))
        lastSeqno = int(sortedFiles[-1].split('-')[1])
    else:
        lastSeqno = -1

    if lcm:
        print("Using LCM")
    if tiny:
        print("Using Tiny VAE")

    print(f"\ngenerating {nBatches*batchSize} images with {nSteps} LCM steps")
    print(f"It can take a few minutes to download the model the very first run.")
    print("After that it can take 10's of seconds to load the rather large sdxl model.")

    seed = random.randint(0, 2147483647)
    torch.manual_seed(seed)

    genit(pipe, guidance, prompt, nBatches, batchSize, nTokens, nSteps)
