# Text2SVG

## Installation

Create a new conda environment:

```shell
conda create --name svg_dream python=3.10
conda activate svg_dream
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install shapely 
pip install Pillow==9.5.0 scikit-image==0.19.3 opencv-python matplotlib
pip install numpy scipy timm scikit-fmm einops scikit-learn
pip install accelerate transformers safetensors datasets 
pip install cairosvg rembg pycpd easydict munch kornia
pip install faiss-cpu pytorch_metric_learning fast_pytorch_kmeans
pip install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
```

Install CLIP:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Install diffusers:

```shell
pip install diffusers
```

Install diffvg:

```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
cd ..
cp -R ./upd_pydiffvg/* ~/anaconda3/envs/svg_dream/lib/python3.10/site-packages/diffvg-0.0.1-py3.10-linux-x86_64.egg/pydiffvg/
rm -rf pydiffvg
```

Install pypotrace:

```shell
sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config

git clone -b to_xml https://github.com/mehdidc/pypotrace.git
cd pypotrace
pip install .
cd ..
rm -rf pypotrace
```

Note: The repository is still in progress, and the code needs to be optimized for clarity and simplicity.

## Text to SVG with VSD

```
python t2svg_vsd.py
```

## SVG Optimization with Image Guidance

```
python img_guidance.py
```
