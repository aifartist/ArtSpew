import time
import random
import os, signal
import subprocess
from subprocess import check_output

import torch
from diffusers import DiffusionPipeline

doCompile = False
# For Linux: Use at your own risk.
goCrazy = False

torch.set_default_device('cuda')
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0

pipe = DiffusionPipeline.from_pretrained('SimianLuo/LCM_Dreamshaper_v7', custom_pipeline='latent_consistency_txt2img', scheduler=None, safety_checker=None)
pipe.to(torch_device="cuda", torch_dtype=torch.float16)
pipe.unet.to(memory_format=torch.channels_last)

from diffusers import AutoencoderTiny
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

if doCompile:
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
    pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')
    pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
    pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

guidance = 10.0

if doCompile:
    # Warmup
    print('\nStarting warmup of two images.  If using compile()')
    print('    this can take an extra 35 seconds each time.')
    with torch.inference_mode():
        img = pipe(prompt='The cat in the hat', width=512, height=512,
            num_inference_steps=4,
            guidance_scale=guidance, lcm_origin_steps=50,
            output_type="pil", return_dict=False)
        img = pipe(prompt='The cat in the hat', width=512, height=512,
            num_inference_steps=4,
            guidance_scale=guidance, lcm_origin_steps=50,
            output_type="pil", return_dict=False)

def genit(nImage, nSteps):
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

                pe = torch.zeros((1, 77, 768))
                if False:
                    for i in range(8000):
                        m = random.randint(0, 77-1)
                        n = random.randint(0, 768-1)
                        pe[0][m][n] = 1. * numpy.random.random() - .5

                    for i in range(800):
                        m = random.randint(0, 77-1)
                        n = random.randint(0, 768-1)
                        pe[0][m][n] = 4 * numpy.random.random() - 2

                    for i in range(80):
                        m = random.randint(0, 77-1)
                        n = random.randint(0, 768-1)
                        pe[0][m][n] = 16 * numpy.random.random() - 8

                    for i in range(8):
                        m = random.randint(0, 77-1)
                        n = random.randint(0, 768-1)
                        pe[0][m][n] = 64 * numpy.random.random() - 32
                else:
                    r = torch.normal(mean=0, std=4., size=(4096,))
                    for x in r:
                        m = random.randint(0, 77-1)
                        n = random.randint(0, 768-1)
                        pe[0][m][n] = x


                tm00 = time.time()
                img = pipe(prompt=None, width=512, height=512,
                    num_inference_steps=nSteps,
                    prompt_embeds = pe,
                    guidance_scale=guidance, lcm_origin_steps=50,
                    output_type="pil", return_dict=False)[0]
                print(f"time = {(1000*(time.time() - tm00)):5.2f} milliseconds")
                img[0].save(f"spew/spew-{nSteps}-{idx:09d}.jpg")
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
    if not os.path.exists('spew'):
        os.makedirs('spew')

    if len(sys.argv) - 1 == 0:
       print('You must specify the count of images to generate')
       sys.exit(-1)

    nImages = int(sys.argv[1])

    if len(sys.argv) - 1 == 2:
        nSteps  = int(sys.argv[2])
    else:
        nSteps = 4

    print(f"generating {nImages} with {nSteps} LCM steps")
    genit(nImages, nSteps)
