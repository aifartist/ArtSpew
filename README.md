# ArtSpew
An infinite number of monkeys randomly throwing paint at a canvas

## Explore the diversity of Stable Diffusion through random prompt embeddings.
Using randomly generated token id's this app can generate a huge diverse set of images very rapidly.  I've created 1000 images in 34.6 seconds on a 4090.  That is not a typo.  4 step LCM, 512x512, batchsize=8, using torch.compile.  With 1024x1024 batchsize=2 sdxl you can get great images in under .46 seconds per image

Tongue in cheek:
In ancient granite Wikipedia carvings dated near the end of August 2022 Stable Diffusion was inscribed and human artists were no more. They were replaced by AI Artists typing words, getting the spelling right 2 out of 3 times and usually finding the "Generate" button.  Now, eons later, we are at a point where AI Artists will soon be replaced.  The dawn of ArtSpew is at hand.  Harnessing the magic of a random number generator you can excrete 1000 random images very quickly.

## Installation

### Linux

```
# Get the code
git clone https://github.com/aifartist/ArtSpew
cd ArtSpew

# Create a virtual environment in the current directory
python3 -m venv ./venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip3 install -r requirements.txt
```

## How to spew art with no discernible talent.  :-)

### As a CLI

Try the following for a first run:
```
python3 artspew.py --prompt "crazy cat"
```

It should be super fast. 16 to 29 images per second on a 4090 depending on the number of steps and other options. Use --xl to use the XL model instead of 1.5.

Use:  `python3 artspew.py --help`
to get the full usage.

The new images with appear in the directory named **spew/**.  Even at nSteps=4 there'll be some good ones.  The idea is to see the space of possibilities and generate ideas.  Set nSteps to 12 or 16 and you'll get better images.  Then perhaps set "-c" to 100 and look at the diversity of images you get.

Then try to pick a subject like "Mutant kitty monsters with glowing eyes, H.R.Giger style" and gen random versions of it.  THe command below is doing 10 inference steps and appending 5 random tokens to the prompt.

> python3 artspew.py --xl -p "Mutant kitty monsters with glowing eyes, H.R.Giger style" -c 32 -b 2 -s 10 -r 5 --lcm --tiny-vae -g 0

### As a python module

```
from artspew import ArtSpew
artspew = ArtSpew()
images = artspew.sd.generate_images("crazy cat")
```

Images will contain a list of PIL images wrapped in our Image class.  You can save them to disk with:

```
for idx, image in enumerate(images):
    image.save(f"{idx}.jpg")
```

![image](https://github.com/aifartist/ArtSpew/assets/116415616/f80a5cd9-994f-4134-8e05-f735116bce53)
