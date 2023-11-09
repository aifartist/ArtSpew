# ArtSpew
An infinite number of monkeys randomly throwing paint at a canvas

### Explore the diversity of Stable Diffusion through random prompt embeddings.
Using randomly generated embeddings this app can generate a huge diverse set of images very rapidly.  I've created 1000 images in 133 seconds and haven't even begun to optimize.  With 4 step LCM images I should be able to get to 60 seconds and with batching and torch.compile() under 40 seconds.

In ancient granite Wikipedia carvings dated near the end of August 2022 Stable Diffusion was inscribed and human artists were no more. They were replaced by AI Artists typing words, getting the spelling right 2 out of 3 times and usually finding the "Generate" button.  Now, eons later, we are at a point where AI Artists will soon be replaced.  The dawn of ArtSpew is at hand.  Harnessing the magic of a random number generator you can excrete 1000 random images very quickly.

As it was in the beginning, quality will improve over time.

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
### How to spew art with no discernible talent.
```
python3 artspew.py <desired image count> <desired quality from 1 to 20.  Default 4>
```
The new images with appear in the directory named **spew/**.  The images are creatively named spew-X-Y.jpg where X is the quality value(number of steps) and Y is a sequence number.  Perhaps 30% of the images are noise. Even at quality=4 there'll be some good ones.  The idea is to see the space of possibilities and generate ideas.  Set quality to 12 or 16 and you'll get more good images.  It is a complex process to figure out the ideal internal settings to optimize the percentage of good images.  I'm working on this but try to generate 100 images at quality 12 and look at the variety.  Image using one of the better sd1.5 models with this.  I'll do that soon.

![image](https://github.com/aifartist/ArtSpew/assets/116415616/f80a5cd9-994f-4134-8e05-f735116bce53)
