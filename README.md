# <div align="center" id="toolkit">Generative Model Inference Toolkit (Off-the-Shelf)🎨</div>

<div align="center">
  <a href="http://dlvc-lab.net/lianwen/"> <img alt="SCUT DLVC Lab" src="https://img.shields.io/badge/SCUT-DLVC_Lab-A85882?logo=Academia&logoColor=hsl"></a>
  <a href="./LICENSE"> <img alt="Static Badge" src="https://img.shields.io/badge/License-Apache2.0-FFBF00?&logoColor=rgb&labelColor=gray"></a>
<p></p>

</div>

This is a collection of an off-the-shelf version inference code for mainstream generative models, such as Qwen-Image, BAGEL. The inference supports single/multiple GPUs, both T2I generation and image editing, which is re-formated from the official code or other open-source code by myself.

## <div align="center" id="qwen-image">Qwen-Image</div>

### :hammer_and_pick:Features

`qwen-image/inference.py`: A unified interface that implements T2I generation and single (multi)-image editing using Qwen-Image model, with multi-GPU support.

The Qwen-Image team merely provide image generation code without providing the image editing code. This function is implemented by non-official developers which has now been incorporated in the [latest `diffusers` library](https://github.com/huggingface/diffusers/issues/12065). This `inference.py` in code incorporates the official T2I generation code and the non-official image editing code into one class, supporting easy, off-the-shelf inference and modification.

- [x] **Multiple GPU Inference**. *Notes*: The model can be loaded on three < 24G GPUs, such as RTX 3090 or RTX 4090. If you have two 48G GPUs, you can substitute all the `cuda:2` with`cuda:0` or `cuda:1`.
- [x] **Image Generation**.
- [x] **Single-Image Editing**.
- [x] **Multi-Image Editing**. 

### :camping:Demo

T2I generation

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python inference.py --generate
```

Single image editing

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python inference.py --single_edit
```

Multi-image editing

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python inference.py --multi_edit
```

## <div align="center" id="qwen-image">BAGEL</div>

### :hammer_and_pick:Features

`bagel/generate.py`: Multi-GPU inference code for T2I generation.

`bagel/edit.py`: Multi-GPU inference for code for instruction-based image editing.

- [x] **Multiple GPU Inference**. *Notes*: Implement GPU load balancing using `accelerate` library, validated automatic configuration of two 48GB GPUs.
- [x] **Image Generation**.
- [x] **Single-Image Editing**.
- [x] **Multi-Image Editing**. 

### :camping:Demo

T2I generation

```bash
CUDA_VISIBLE_DEVICES=0,1 python generate.py --model_path /path/to/your/model --prompt 'Generate a cat.'
```

Image editing

```bash
CUDA_VISIBLE_DEVICES=0,1 python edit.py --model_path /path/to/your/model --prompt 'Change the cat to a dog.' --input_image ./asset/cat.jpg --think
```

## <div align="center">:pager:Cotact</div>

Peirong Zhang: eeprzhang@mail.scut.edu.cn

## <div align="center">:beginner:Acknowledgement</div>

[Qwen-Image Image Editing](https://github.com/huggingface/diffusers/issues/12065), [Multi-GPU Inference for Qwen-Image](https://paste.ubuntu.com/p/b7ddVMQ2q8/)
