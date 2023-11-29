import random
import os
import argparse
import torch
import logging
from src.stable_diffusion_sd15 import StableDiffusionSD15
from src.stable_diffusion_sdxl import StableDiffusionSDXL
from pathvalidate import sanitize_filename

MODEL_ID_SD15 = 'runwayml/stable-diffusion-v1-5'
MODEL_ID_SDXL = 'stabilityai/stable-diffusion-xl-base-1.0'
DEFAULT_N_RANDOM_TOKENS = 5
DEFAULT_BATCH_COUNT = 10
DEFAULT_BATCH_SIZE = 2
DEFAULT_LCM = True
DEFAULT_SEED = -1
DEFAULT_TINY_VAE = False
DEFAULT_TORCH_COMPILE = False

DEFAULT_STEPS = -1
DEFAULT_SD_STEPS = 20
DEFAULT_LCM_STEPS = 8

DEFAULT_GUIDANCE = -1.
DEFAULT_GUIDANCE_SD = 8.
DEFAULT_GUIDANCE_LCM = 0.

DEFAULT_WIDTH = -1
DEFAULT_HEIGHT = -1
DEFAULT_WIDTH_SD15 = 512
DEFAULT_HEIGHT_SD15 = 512
DEFAULT_WIDTH_SDXL = 1024
DEFAULT_HEIGHT_SDXL = 1024


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--xl', action=argparse.BooleanOptionalAction,
                        help='Use SDXL')
    parser.add_argument('-m', '--model-id', type=str, default='auto',
                        help='Specify the input file')
    parser.add_argument('-p', '--prompt', type=str,
                        help='Specify the start of the prompt')
    parser.add_argument('-w', '--width', type=int, default=DEFAULT_WIDTH,
                        help='Image width, -1 for auto')
    parser.add_argument('-h', '--height', type=int, default=DEFAULT_HEIGHT,
                        help='Image height, -1 for auto')
    parser.add_argument('-n', '--number-of-images', type=int, default=DEFAULT_BATCH_COUNT,
                        help='Number of batches to do')
    parser.add_argument('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch Size')
    parser.add_argument('-s', '--n_steps', type=int, default=DEFAULT_STEPS,
                        help='Number of inference steps, -1 for auto')
    parser.add_argument('-r', '--random-tokens', type=int, default=DEFAULT_N_RANDOM_TOKENS,
                        help='Number of random tokens added')
    parser.add_argument('--lcm', action='store_true',
                        help='Use LCM')
    parser.add_argument('--tiny-vae', action='store_true',
                        help='Use the tiny VAE')
    parser.add_argument('-g', '--guidance', type=float, default=DEFAULT_GUIDANCE,
                        help='Guidance value, -1 for auto')
    parser.add_argument('--torch-compile', action='store_true',
                        help='Using torch.compile for faster inference')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debug info')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Only show errors')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed, -1 for random')
    
    args = parser.parse_args()
    return args


class ArtSpew:

    SD15 = 'sd15'
    SDXL = 'sdxl'

    def __init__(self, **kwargs):
        self.sd = None
        self.model_type = None

        self._xl = kwargs.pop('xl', False)
        model_id = kwargs.pop('model_id', 'auto')
        tiny_vae = kwargs.pop('tiny_vae', DEFAULT_TINY_VAE)
        lcm = kwargs.pop('lcm', DEFAULT_LCM)
        width = kwargs.pop('width', DEFAULT_WIDTH)
        height = kwargs.pop('height', DEFAULT_HEIGHT)
        seed = kwargs.pop('seed', DEFAULT_SEED)
        batch_count = kwargs.pop('batch_count', DEFAULT_BATCH_COUNT)
        batch_size = kwargs.pop('batch_size', DEFAULT_BATCH_SIZE)
        if seed == DEFAULT_SEED:
            seed = random.randint(0, 2147483647)
        n_random_tokens = kwargs.pop('n_random_tokens', DEFAULT_N_RANDOM_TOKENS)
        n_steps = kwargs.pop('n_steps', DEFAULT_STEPS)
        guidance_scale = kwargs.pop('guidance_scale', DEFAULT_GUIDANCE)
        torch_compile = kwargs.pop('torch_compile', DEFAULT_TORCH_COMPILE)

        if len(kwargs) > 0:
            raise ValueError(f"Unknown arguments: {kwargs}")

        if model_id == 'auto':
            model_id = MODEL_ID_SDXL if self._xl else MODEL_ID_SD15
        if width == -1:
            width = DEFAULT_WIDTH_SDXL if self._xl else DEFAULT_WIDTH_SD15
        if height == -1:
            height = DEFAULT_HEIGHT_SDXL if self._xl else DEFAULT_HEIGHT_SD15
        if n_steps == -1:
            n_steps = DEFAULT_LCM_STEPS if lcm else DEFAULT_SD_STEPS
        if guidance_scale == -1:
            guidance_scale = DEFAULT_GUIDANCE_LCM if lcm else DEFAULT_GUIDANCE_SD

        self.model_type = self._detect_model_type(model_id)

        if self.model_type == self.SD15:
            sd_class = StableDiffusionSD15
        elif self.model_type == self.SDXL:
            sd_class = StableDiffusionSDXL
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.sd = sd_class(
            model_id=model_id,
            tiny_vae=tiny_vae,
            lcm=lcm,
            width=width,
            height=height,
            seed=seed,
            batch_count=batch_count,
            batch_size=batch_size,
            n_random_tokens=n_random_tokens,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            torch_compile=torch_compile
        )

    def _detect_model_type(self, model_id):
        # Examine the model to determine the model type?
        # For now, just read the xl argument.
        if self._xl:
            return self.SDXL
        else:
            return self.SD15


def main():
    args = parse_arguments()

    if not os.path.exists('spew'):
        os.makedirs('spew')

    # Setup Logger.
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ArtSpew")

    logger.info(
        f"Generating {args.batch_count*args.batch_size} images with {args.steps} steps. "
        "It can take a few minutes to download the model the very first run. "
        "After that it can take 10s of seconds to load the stable diffusion model. "
    )

    artspew = ArtSpew(
        xl=args.xl,
        model_id=args.model_id,
        tiny_vae=args.tiny_vae,
        lcm=args.lcm,
        width=args.width,
        height=args.height,
        seed=args.seed,
        batch_count=args.batch_count,
        batch_size=args.batch_size,
        n_random_tokens=args.random_tokens,
        n_steps=args.steps,
        guidance_scale=args.guidance,
        torch_compile=args.torch_compile
    )

    sequence_number = -1
    if not os.path.exists('spew'):
        os.makedirs('spew')

    files = [entry.name for entry in os.scandir('spew') if entry.name.startswith(artspew.sd.get_filename_prefix())]

    if files:
        sorted_files = sorted(files, key=lambda x: int(x.split('-')[1]))
        sequence_number = int(sorted_files[-1].split('-')[1])

    images = artspew.sd.generate_images(args.prompt)
    for idx, image in enumerate(images):
        sequence_number += 1
        safe_prompt = sanitize_filename(image.prompt_text)
        image.save(f"spew/{artspew.sd.get_filename_prefix()}{sequence_number:09d}-{idx:02d}-{safe_prompt}.jpg")


if __name__ == "__main__":
    main()
