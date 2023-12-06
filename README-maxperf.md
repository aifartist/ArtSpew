This demos close to 150 images per second on a 4090.
I can do faster but this should be good enough to prove
the perf I can get.

Presuming you have ArtSpew installed:

    pip3 install -r req-maxperf.txt
    pip3 install -r req-sfast.txt

Use the following brings up the GUI.
Use the step, go, and stop buttons until your eyes go bleed.

    python3 maxperf.py <batchsize:max=10>

If a batch size of 10 causes an OOM reduce the number.
