# ArtSpew
An infinite number of monkeys randomly throwing paint at a canvas

### Explore the diversity of Stable Diffusion through random prompt embeddings.

In anchient granite Wikipedia carvings dated near the end of August 2022 Stable Diffusion was inscribed and human artists were no more. They were replaced by AI Artists typing words, getting the spelling right 2 out of 3 times and usually finding the "Generate" button.  Now, eons later, we are at a point where AI Artists will soon be replaced.  The dawn of ArtSpew is at hand.  Harnessing the vast power of a random number generator you can excrete 1000 random images in 60 seconds with a good GPU.  45 seconds if the wind is blowing at your back(torch.compile()).  And soon, when I adding batching, it'll be even faster.

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
python3 artspew.py <desired image count> <desired quality from 1 to 20>
```
The new images with appear in the directory named **spew/**.  The images are creatively named spew-X-Y.jpg where X is the quality value(number of steps) and Y is a sequence number.
