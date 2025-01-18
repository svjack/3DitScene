# 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://zqh0253.github.io/3DitScene/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/qihang/3Dit-Scene/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.18424-b31b1b.svg)](https://arxiv.org/abs/2405.18424) 


<table class="center">
    <tr style="line-height: 0">
      <td width=35% style="border: none; text-align: center">Move the bear, and rotate the camera</td>
      <td width=30% style="border: none; text-align: center">Move / remove the girl, and rotate the camera</td>
    </tr>
    <tr style="line-height: 0">
      <td width=35% style="border: none"><img src="assets/bear.gif"></td>
      <td width=30% style="border: none"><img src="assets/cherry.gif"></td>
    </tr>
 </table>

### 1. 系统要求
- 操作系统：Ubuntu 22.04
- GPU：支持 CUDA 的 NVIDIA GPU（建议使用 CUDA 11.8）
- 用户：建议使用非 root 用户进行操作

---

### 2. 安装系统依赖项
更新系统并安装必要的依赖项：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    zip \
    unzip
```

---

### 3. 安装 Miniconda（如果尚未安装）
如果系统中没有安装 `conda`，可以通过以下步骤安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

按照提示完成安装后，重新加载 shell 配置：

```bash
source ~/.bashrc
```

---

### 4. 创建并激活 Conda 虚拟环境
创建一个名为 `py310` 的虚拟环境，并激活它：

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create -n py310 python=3.10 -y
conda activate py310
```

安装 `ipykernel` 并将虚拟环境添加到 Jupyter Notebook 内核中（可选）：

```bash
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"
```

---

### 5. 配置 CUDA 环境（可选）
如果你已经安装了 CUDA，可以配置相关的环境变量：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
```

---

### 6. 安装 Python 依赖项
升级 `pip` 和 `setuptools`，并安装 `ninja`：

```bash
pip install --upgrade pip setuptools==69.5.1 ninja
```

#### 安装 PyTorch 和 TorchVision（不指定 CUDA 版本）
```bash
pip install xformers==0.0.22 torch torchvision
```

#### 安装 PyTorch 和 TorchVision（指定 CUDA 11.8 版本）
```bash
pip install xformers==0.0.22 torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

#### 安装 `nerfacc`（不指定 CUDA 版本）
```bash
pip install nerfacc
```

#### 安装 `nerfacc`（指定 CUDA 11.8 版本）
```bash
pip install nerfacc==0.5.2 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html
```

---

### 7. 安装项目依赖项
克隆项目仓库并安装依赖项：

```bash
git clone https://github.com/svjack/3DitScene.git ~/threestudio --recursive
cd ~/threestudio
pip install -r requirements.txt
pip install -U fastapi pydantic
```

---

### 8. 下载并安装额外的依赖项
下载并解压 `cache.zip`：

```bash
wget --quiet https://www.dropbox.com/scl/fi/2s4b848d4qqrz87bbfc2z/cache.zip?rlkey=f7tyf4952ey253xlzvb1lwnmc -O tmp.zip
unzip tmp.zip
```

安装 `segment-anything-langsplat` 和 `MobileSAM-lang`：

```bash
pip install ./submodules/segment-anything-langsplat
pip install ./submodules/MobileSAM-lang
```

#### 下载并安装 `diff_gaussian_rasterization`（不指定 CUDA 版本）
```bash
pip install ./submodules/diff-gaussian-rasterization
```

#### 下载并安装 `diff_gaussian_rasterization`（指定 CUDA 11.8 版本）
```bash
wget --quiet https://www.dropbox.com/scl/fi/rhl1r9qww9fq6jtjmh43x/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl?rlkey=xp02kfjvyk9urnacybp4ll108 -O diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
pip install diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
```

安装 `simple-knn`：

```bash
pip install ./submodules/simple-knn
```

---

### 9. 下载模型权重
创建 `ckpts` 目录并下载模型权重：

```bash
mkdir ckpts
wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth
cp ~/threestudio/submodules/MobileSAM-lang/weights/mobile_sam.pt ./ckpts/
```

---

### 10. 运行项目
在 `py310` 虚拟环境中运行项目：

```bash
python gradio_app_single_process.py --listen --hf-space
```

---

### 11. 环境变量配置（可选）
如果你有特定的 GPU 架构，可以通过设置以下环境变量来加速构建过程：

```bash
export TORCH_CUDA_ARCH_LIST="8.6"  # 例如 RTX 30xx 系列
export TCNN_CUDA_ARCHITECTURES=86
```

---

### 12. 清理缓存（可选）
清理不必要的缓存以节省空间：

```bash
rm -rf ~/.cache
```

## Installation

+ Install `Python >= 3.8`.
+ Install `torch >= 1.12`. We have tested on `torch==2.0.1+cu118`, but other versions should also work fine.
+ Clone our repo:
```
git clone https://github.com/zqh0253/3DitScene.git --recursive
```
+ Install dependencies:
```
pip install -r requirements.txt
```
+ Install submodules:
```
pip install ./submodules/segment-anything-langsplat
pip install ./submodules/MobileSAM-lang
pip install ./submodules/langsplat-rasterization
pip install ./submodules/simple-knn
```
+ Prepare weights for `SAM`:
```
mkdir ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth
cp submodules/MobileSAM-lang/weights/mobile_sam.pt ./ckpts/
```

## Usage

Run the following command to launch the optimization procedure: 
```
python -u launch.py --config custom/threestudio-3dgs/configs/scene_lang.yaml  --train --gpu 0 tag=3DitScene 
system.geometry.geometry_convert_from=depth:${IMGPATH} system.geometry.ooi_bbox=${BBOX}
system.prompt_processor.prompt="${PROMPT}" system.empty_prompt="${EMPTY_PROMPT}" system.side_prompt="${SIDE_PROMPT}"
```
You should specify the image path `IMGPATH`, the bounding box of the interested object  `BBOX`, and the promtps: `PROMPT`, `EMPTY_PROMPT`, `SIDE_PROMPT`. These prompts describe the image itself, the background area behind the image, and the content of the novel view region, respectively.

Here we provide an image (`./assets/teddy.png`) as example:
```
python -u launch.py --config custom/threestudio-3dgs/configs/scene_lang.yaml  --train --gpu 0 tag=3DitScene 
system.geometry.geometry_convert_from=depth:assets/teddy.png system.geometry.ooi_bbox=[122,119,387,495]
system.prompt_processor.prompt="a teddy bear in Times Square" system.empty_prompt="Times Square, out of focus" system.side_prompt="Times Square, out of focus"
```

## Huggingface demo

We provide a huggingface demo. You have two options to explore our demo: 
(1) Visit our [online Hugging Face space](https://huggingface.co/spaces/qihang/3Dit-Scene).
(2) Deploy it locally by following these steps:
+ Install the necessary packages and download required files as specified in our [Dockerfile](https://huggingface.co/spaces/qihang/3Dit-Scene/blob/main/Dockerfile),
+ Run the following command to launch the service at `localhost:10091`:
```
python gradio_app_single_process.py --listen --hf-space --port 10091
```

## Citation

If you find our work useful, please consider citing:
```
inproceedings{zhang20243DitScene,
  author = {Qihang Zhang and Yinghao Xu and Chaoyang Wang and Hsin-Ying Lee and Gordon Wetzstein and Bolei Zhou and Ceyuan Yang},
  title = {{3DitScene}: Editing Any Scene via Language-guided Disentangled Gaussian Splatting},
  booktitle = {arXiv},
  year = {2024}
}
```
