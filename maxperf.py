import sys
import PIL
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen, QFont, QPainter
from PyQt5.QtCore import Qt, QTimer, QEvent, pyqtSignal, QCoreApplication

import numpy as np

import torch
from diffusers import AutoPipelineForText2Image
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

mw = None
batchSize = 10
prompts = ['Evil space kitty', 'Cute dog in hat, H.R. Giger style', 'Horse wearing a tie', 'Cartoon pig', 'Donkey on Mars', 'Cute kitties baked in a cake', 'Boxing chickens on farm, Maxfield Parish style', 'Future spaceship', 'A city of the past', 'Jabba the Hut wearing jewelery']

def dwencode(pipe, prompts, batchSize: int, nTokens: int):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    if nTokens < 0 or nTokens > 75:
        raise BaseException("n random tokens must be between 0 and 75")

    if nTokens > 0:
        randIIs = torch.randint(low=0, high=49405, size=(batchSize, nTokens), device='cuda')

    text_inputs = tokenizer(
        prompts,
        padding = "max_length",
        max_length = tokenizer.model_max_length,
        truncation = True,
        return_tensors = "pt",
    ).to('cuda')

    tii = text_inputs.input_ids

    if nTokens > 0:
        for i in range(batchSize):
            # Find the end mark which is deterimine the
            # prompt len(pl) in terms of user tokens.
            pl = (tii[i] == torch.tensor(49407, device='cuda')).nonzero()[0][0].item() - 1
            tii[i][1+pl:1+pl+nTokens] = randIIs[i]
            tii[i][1+pl+nTokens] = 49407

    if False:
        for bi in range(batchSize):
            pl = (tii[i] == torch.tensor(49407, device='cuda')).nonzero()[0][0].item() - 1
            print(f"{mw.seqno:05d}-{bi:02d}: ", end='')
            for tid in tii[bi][1:1+pl+nTokens]:
                print(f"{tokenizer.decode(tid)} ", end='')
            print('')

    prompt_embeds = text_encoder(tii.to('cuda'), attention_mask=None)
    prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device='cuda')

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

    return prompt_embeds


pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False
)
pipe.to("cuda")
#pipe.unet.to(memory_format=torch.channels_last)

from diffusers import AutoencoderTiny
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

pipe.set_progress_bar_config(disable=True)

if True:
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # It might be slow for Triton to generate, compile and fine-tune kernels.
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    # NOTE:
    # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # Disable Triton if you encounter this problem.
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    # NOTE:
    # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    # My implementation can handle dynamic shape with increased need for GPU memory.
    # But when your GPU VRAM is insufficient or the image resolution is high,
    # CUDA Graph could cause less efficient VRAM utilization and slow down the inference,
    # especially when on Windows or WSL which has the "shared VRAM" mechanism.
    # If you meet problems related to it, you should disable it.
    config.enable_cuda_graph = True

    if True:
        config.enable_jit = True
        config.enable_jit_freeze = True
        config.trace_scheduler = True
        config.enable_cnn_optimization = True
        config.preserve_parameters = False
        config.prefer_lowp_gemm = True

    pipe = compile(pipe, config)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.lasttm = time.time()

        self.ii = 0
        self.seqno = 0
        self.stopped = True

        font = QFont("Arial", 24)
        self.fps = QLineEdit(self)
        self.fps.setFixedWidth(176)
        self.fps.setFont(font)
        self.seed = QLineEdit(self)
        self.seed.setText("  Ultra fast RTSD by Daniel Wood aka AIFartist")
        self.seed.setFont(font)

        self.go = QPushButton('Go', self)
        self.step = QPushButton('Step', self)
        self.stop = QPushButton('Stop', self)

        self.nImgs = 10
        # Create the image areas
        self.imgs = []
        for ii in range(self.nImgs):
            self.imgs.append(QtWidgets.QLabel())
            self.imgs[ii].setFixedSize(512, 512)

        # Layout the widgets
        layout = QVBoxLayout()

        l2 = QHBoxLayout()
        l2.addWidget(self.fps)
        l2.addWidget(self.seed)
        l2.addWidget(self.go)
        l2.addWidget(self.step)
        l2.addWidget(self.stop)
        layout.addLayout(l2)

        imgl = QGridLayout()
        for ii in range(self.nImgs):
            row = ii // 5
            col = ii % 5
            imgl.addWidget(self.imgs[ii], row, col)
        layout.addLayout(imgl)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.do_event)
        self.timer_interval = 0

        self.setLayout(layout)

        self.go.clicked.connect(self.do_go)
        self.step.clicked.connect(self.do_step)
        self.stop.clicked.connect(self.do_stop)

        self.genImage()

    def post_button_click_event(self):
        event = QEvent(QEvent.Type(QEvent.MouseButtonPress))
        QCoreApplication.postEvent(self.go, event)

    def do_event(self):
        if not self.stopped:
            self.do_go()

    def do_stop(self):
        self.stopped = True

    def do_step(self):
        if self.stopped:
            self.genImage()

    def do_go(self):
        global batchSize
        self.stopped = False
        self.genImage()
        tm = time.time()
        self.fps.setText(f"{(batchSize/(tm-self.lasttm)):5.1f} fps")
        print(f"time={(1000.*(tm-self.lasttm)):3.1f}ms")
        self.lasttm = tm
        self.timer.start(self.timer_interval)

    def genImage(self):
        global prompts, batchSize
        seed = random.randint(0, 2147483647)
        torch.manual_seed(seed)

        images = genit(0, prompts=prompts, batchSize=batchSize, nSteps=1)
        for img in images:
            imgData = img.tobytes('raw', 'RGB')
            qImg = QImage(imgData, 512, 512, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            painter = QPainter(pixmap)
            font = QFont()
            font.setPointSize(32)
            painter.setPen(QColor(255, 255, 0))
            painter.setFont(font)
            painter.drawText(24, 64, f"{self.seqno:4d}")
            painter.end()
            self.imgs[self.ii].setPixmap(pixmap)
            self.ii += 1
            if self.ii == self.nImgs:
                self.ii = 0
            self.seqno += 1

import time
import random
import torch

def genit(mode, prompts, batchSize, nSteps):
    #tm0 = time.time()
    pe = dwencode(pipe, prompts, batchSize, 9)
    images = pipe(
        prompt_embeds = pe,
        width=512, height=512,
        num_inference_steps = nSteps,
        guidance_scale = 1,
        output_type="pil",
        return_dict=False
    )[0]
    #print(f"time = {(1000*(time.time() - tm0)):3.1f} milliseconds")

    return images

if __name__ == '__main__':
    if len(sys.argv) == 2:
        batchSize = int(sys.argv[1])
        if batchSize > 10:
            print('Batchsize must not be greater than 10.')
        prompts = prompts[:batchSize]
    else:
        batchSize = 10
        prompts = ['Evil space kitty', 'Cute dog in hat, H.R. Giger style', 'Horse wearing a tie', 'Cartoon pig', 'Donkey on Mars', 'Cute kitties baked in a cake', 'Boxing chickens on farm, Maxfield Parish style', 'Future spaceship', 'A city of the past', 'Jabba the Hut wearing jewelery']
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
