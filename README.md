[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Implement UNet Architecture from scratch
- Use OxfordIIITPet dataset for training
- Give option to use MaxPooling or Strided Convolution for transition block
- Give option to use Transpose Convolution or Upsample for up convolution
- Give optim to Cross Entropy or Dice for loss calculation

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]

## :open_file_folder: Files
- [**dataset.py**](dataset.py)
    - This file contains class to load and process data
- [**model.py**](model.py)
    - This file contains model architecture
- [**utils.py**](utils.py)
    - This file contains methods to plot metrics and results.
- [**train.py**](train.py)
    - This is the main file of this project
    - It uses function available in `dataset.py`, `model.py` and `utils.py`
    - It contrains functions to train and test model.

## :building_construction: Model Architecture
The model is implemented based on U-Net: Convolutional Networks for Biomedical Image Segmentation.
There are five encoder blocks each return output of before and after applying transition. Thre four
decoder blocks using inputs from previout layer and corresponding block of encoder part. 

## :golfing: Training Options

- Transposed Convolutional: Instead of sliding the kernel over the input and performing element-wise multiplication and summation, a transposed convolutional layer slides the input over the kernel and performs element-wise multiplication and summation.
- Upsample: Provided tensor is upsampled based on mode.
- Dice Loss: Dice loss focuses on maximising the intersection area over foreground while minimising the Union over foreground.


## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/unet_pytorch
```
2. Go inside folder
```
 cd unet_pytorch
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python train.py

# To train using Max Pooling, Transpose Convolution and Cross Entropy Loss
python train.py --epochs=25 --batch_size=64 --max_pool --transpose_conv --cross_entropy_loss

# To train train from jupyter notebook using Max Pooling, Transpose Convolution and Dice Loss and visulize plots 
%run train.py --epochs=25 --batch_size=128 --max_pool --transpose_conv

```

## Usage 
Please refer to [ERA V2 Session 22](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-22)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
* [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [U-NET Implementation from Scratch using TensorFlow](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/