import time
import random
import os, signal
import subprocess
from subprocess import check_output

import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

doCompile = False
goCrazy = False # For Linux: Use at your own risk.
lastSeqno = -1

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

from diffusers import AutoencoderTiny
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

if doCompile:
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
    #pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')
    pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
    pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

guidance = 10.0

if doCompile:
    # Warmup
    print('\nStarting warmup of two images.  If using compile()')
    print('    this can take an extra 35 seconds each time.')
    with torch.inference_mode():
        img = pipe(prompt='The cat in the hat', width=1024, height=1024,
            num_inference_steps=12,
            guidance_scale=0, lcm_origin_steps=50,
            output_type="pil", return_dict=False)
        img = pipe(prompt='The cat in the hat', width=1024, height=1024,
            num_inference_steps=12,
            guidance_scale=0, lcm_origin_steps=50,
            output_type="pil", return_dict=False)

def genit(nImage, nSteps):
    global lastSeqno
    with torch.inference_mode():
        try:
            # This STOP'ing of processes idles enough cores to allow
            # one core to hit 5.8 GHz and CPU perf does matter if you
            # have a 4090.
            # Warning: Stopping gnome-shell freezes the desktop until the
            # gen is done.  If you control-c you may have to hard restart.
            if goCrazy:
                subprocess.Popen('pkill -STOP chrome', shell=True)
                pid = int(check_output(["pidof", '-s', '/usr/bin/gnome-shell']))
                os.kill(pid, signal.SIGSTOP)

            tm0 = time.time()
            for idx in range(0, nImages):
                import random
                import numpy

                seed = random.randint(0, 2147483647)
                torch.manual_seed(seed)

                pe = torch.zeros((1, 77, 2048))
                pe.to(device='cuda:0', dtype=torch.float16)
                r = torch.normal(mean=0, std=4., size=(4096,))
                for x in r:
                    m = random.randint(0, 77-1)
                    n = random.randint(0, 2048-1)
                    pe[0][m][n] = x

                ppe = torch.zeros((1, 1280))
                r = torch.normal(mean=0, std=2., size=(1280,))
                for x in r:
                    n = random.randint(0, 1280-1)
                    ppe[0][n] = x

                ppe.to(device='cuda', dtype=torch.float16)
                pe.to(device='cuda', dtype=torch.float16)

                tm00 = time.time()
                img = pipe(prompt=None, width=1024, height=1024,
                    num_inference_steps=nSteps,
                    prompt_embeds = pe,
                    pooled_prompt_embeds = ppe,
                    guidance_scale=0, lcm_origin_steps=50,
                    output_type="pil", return_dict=False)[0]
                print(f"time = {(1000*(time.time() - tm00)):5.2f} milliseconds")
                lastSeqno += 1
                img[0].save(f"spewxl/spew-{lastSeqno:09d}-{nSteps}.jpg")
            print(f"time = {time.time() - tm0}")
        except Exception as e:
            raise e
        finally:
            pass
            if goCrazy:
                os.kill(pid, signal.SIGCONT)
            subprocess.Popen('pkill -CONT chrome', shell=True)

import sys
if __name__ == "__main__":
    if not os.path.exists('spewxl'):
        os.makedirs('spewxl')

    files = os.listdir('spewxl')
    if len(files) > 0:
        sortedFiles = sorted(files, key=lambda x: int(x.split('-')[1]))
        lastSeqno = int(sortedFiles[-1].split('-')[1])
    else:
        lastSeqno = -1

    print(lastSeqno)

    if len(sys.argv) - 1 == 0:
       print('You must specify the count of images to generate')
       sys.exit(-1)

    nImages = int(sys.argv[1])

    if len(sys.argv) - 1 == 2:
        nSteps  = int(sys.argv[2])
    else:
        nSteps = 10

    print(f"generating {nImages} with {nSteps} LCM steps")
    genit(nImages, nSteps)
