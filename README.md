# ArtSpew
An infinite number of monkeys randomly throwing paint at a canvas

## Explore the diversity of Stable Diffusion through random prompt embeddings.
Using randomly generated token id's this app can generate a huge diverse set of images very rapidly.  I've created 1000 images in 34.6 seconds on a 4090.  That is not a typo.  4 step LCM, 512x512, batchsize=8, using torch.compile.  With 1024x1024 batchsize=2 sdxl you can get great images in under .46 seconds per image.  NOTE: This is old data.  Things are even faster now.

Tongue in cheek:
In ancient granite Wikipedia carvings dated near the end of August 2022 Stable Diffusion was inscribed and human artists were no more. They were replaced by AI Artists typing words, getting the spelling right 2 out of 3 times and usually finding the "Generate" button.  Now, eons later, we are at a point where AI Artists will soon be replaced.  The dawn of ArtSpew is at hand.  Harnessing the magic of a random number generator you can excrete 1000 random images very quickly.

## Overview
This repo has the command line program named artspew.py which can generate huge numbers of image for exploration purposes.  Also, including is maxperf.py, a GUI program to wow people with the lastest tech to push things as fast as 150 images per second on a 4090.  After installing ArtSpew look at README-maxperf.md to see how to run that demo.

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
python3 artspew.py --xl --p "crazy cat" -n 10
```

This will generate 100 images of "crazy cat" with random tokens added to increase the diversity of the images. It should be super fast. Lose the "--xl" to use SD 1.5.

Use:  `python3 artspew.py --help`
to get the full usage.

The new images with appear in the directory named **spew/**.  Even at 4 steps there'll be some good ones.  The idea is to see the space of possibilities and generate ideas.  Set steps to 12 or 16 and you'll get better images.  Then perhaps set "-c" to 100 and look at the diversity of images you get.

Then try to pick a subject like "Mutant kitty monsters with glowing eyes, H.R.Giger style" and gen random versions of it.  THe command below is doing 10 inference steps and appending 5 random tokens to the prompt.

> python3 artspew.py --xl -p "Mutant kitty monsters with glowing eyes, H.R.Giger style" -b 2 -s 10

### As a python module

```
from artspew import ArtSpew
artspew = ArtSpew()
image_generator = artspew.create_generator("crazy cat")
idx = -1
for image in image_generator:
    idx += 1
    image.save(f"{idx}.jpg")
```

![image](https://github.com/aifartist/ArtSpew/assets/116415616/f80a5cd9-994f-4134-8e05-f735116bce53)
