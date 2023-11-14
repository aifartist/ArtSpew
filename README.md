# ArtSpew
An infinite number of monkeys randomly throwing paint at a canvas

### Explore the diversity of Stable Diffusion through random prompt embeddings.
Using randomly generated embeddings this app can generate a huge diverse set of images very rapidly.  I've created 1000 images in 133 seconds and haven't even begun to optimize.  With 4 step LCM images I should be able to get to 60 seconds and with batching and torch.compile() under 40 seconds FOR 1000 IMAGES on a 4090.

Tongue in cheek:
In ancient granite Wikipedia carvings dated near the end of August 2022 Stable Diffusion was inscribed and human artists were no more. They were replaced by AI Artists typing words, getting the spelling right 2 out of 3 times and usually finding the "Generate" button.  Now, eons later, we are at a point where AI Artists will soon be replaced.  The dawn of ArtSpew is at hand.  Harnessing the magic of a random number generator you can excrete 1000 random images very quickly.

### Setup Instructions - Linux centric
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
### How to spew art with no discernible talent.  :-)
Try the following for a first run:
```
python3 sdxl.py -l -t -c 8 -b 2 -n 5 -s 12
```
l: use lcm
t: use tiny vae
c: count of batches
b: batch size
n: number of random tokens to use
s: Number of inference steps

Use:  python3 sdxl.py -h
to get the full usage.

The new images with appear in the directory named **spew/**.  Even at nSteps=4 there'll be some good ones.  The idea is to see the space of possibilities and generate ideas.  Set nSteps to 12 or 16 and you'll get better images.  Then perhaps set "-c" to 100 and look at the diversity of images you get.

Also, I know exactly how to substantial speed this up.  This is only the beginning...

![image](https://github.com/aifartist/ArtSpew/assets/116415616/f80a5cd9-994f-4134-8e05-f735116bce53)
