# ArtSpew - Version 0.1.0
An infinite number of monkeys randomly throwing paint at a canvas

## Explore the diversity of Stable Diffusion and generate ideas for art using rapid high volume random image generation.
Whether asking for variations on a theme like "Space kitty cowboy" or just asking for 1000 totally random images, this is your tool.  With SD1.5 models you could get 1000 images in 45 seconds with an NVidia 4090.  We also fully support SDXL.  What will you discover in this universe?

Also, as a pure demonstration of absurd performance I offer ***maxperf*** which can generate 149 images per seconds on a 4090.  How will this very new tech bring about the new reality of realtime videos?  Thing are going to get very interesting soon.

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
### Windows
```
Please contribute Windows install instructions.
```

## How to spew art with no discernible talent.  :-)

Try the following for a first run:
```
python3 artspew.py --xl -p "Space cat cowboy" -c 10 -n 11
```
This will generate 10 images.  Your theme is "Space cat cowboy".  The "-n 11" tells app to add 11 totally random tokens to your prompt to add variety and make the images more interesting.  By using LCM even 1024x1024 generations are much quicker than is typical for SDXL.  Look at the images in your **spew/** directory.  If you have a 4090 just do 100 the first time.

Your next experiment is to get rid of the "-p" prompt and just generate totally random images.  Have patience and do 100.  You should get many interesting images.

Now if you want fast but with reduced quality using the very old sd1.5 default model get rid of the '--xl' and generate at least 100 images with or without a prompt.  On my machine this took 12.45 seconds.  ***Advanced*** users can try the "compile" option that can be found with "--help" and perhaps create 100 images in 6 seconds.  Compile may not work on Windows without WSL(untested).  Another way to get even faster performance is to use "-b" to set the batchsize to generation multiple images in parallel.  On GPU's with small amounts of VRam if you run out of memory decrease the batchsize.

NOTE: Unlike SDXL the base SD 1.5 model doesn't have great quality.  However, there are 1000's of fine tuned or merge SD1.5 models which are much better.  You probably have downloaded a few.  The "-m" option can be used to specify the path to your favorite local SD 1.5 model.  Using --xl but specifying a sd1.5 model with "-m" will likely destroy the entire universe.  Please avoid doing that.

Use:  `python3 artspew.py --help`
to get the full usage.

Where will your imagination take you?
> Mutant kitty monsters with glowing eyes, H.R.Giger style

remember to increase variety by using "-n" to add some random tokens to your prompt.  Use --xl if more quality is desired.  Do more steps using "-s" will also improve the results.

FUTURE: While the "compile" feature adds a lot of speed it is painful to use unless you want to generate a huge number of images. Do not use unless you are familar with torch.compile().  My maxperf mentioned below uses a new compiler technology called stable-fast which is far better.  I will soon integrate it into the artspew program and DOUBLE your speed.

# maxperf
This is a GUI app which is crazy fast just to show how far we've come.  Don't complain about quality of something which came out days ago.  Read the README-maxperf.md for instructions to run it.


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
