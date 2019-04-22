# Fast Style Transfer in Pytorch! 

An implementation of **fast-neural-style** in PyTorch! Style Transfer learns the aesthetic style of a `style image`, usually an art work, and applies it on another `content image`. This repository contains codes the can be used for: 
1. `image-to-video` aesthetic style transfer, and for
2. training `style-learning` transformation network

This implemention follows the style transfer approach outlined in [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) paper by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li*, along with the [supplementary paper detailing the exact model architecture](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf) of the mentioned paper. The idea is to train a **`separate feed-forward neural network (called Transformation Network) to transform/stylize`** an image and use backpropagation to learn its parameters, instead of directly manipulating the pixels of the generated image as discussed in [A Neural Algorithm of Artistic Style aka **neural-style**](https://arxiv.org/abs/1508.06576) paper by *Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge*. The use of feed-forward transformation network allows for fast stylization of images, around 1000x faster than neural style.

The [original caffe pretrained weights of VGG16](https://github.com/jcjohnson/pytorch-vgg) were used for this implementation, instead of the pretrained VGG16's in PyTorch's model zoo.

# Webcam Demo
![Webcam Demo](images/sample.gif)
<p align = 'center'>
</p>


## Requirements
Most of the codes here assume that the user have access to CUDA capable GPU, at least a GTX 1050 ti or a GTX 1060
### Data Files
* [Pre-trained VGG16 network weights](https://github.com/jcjohnson/pytorch-vgg) - put it in `models/` directory
* [Specs Of Faces](https://sites.google.com/view/sof-dataset) - 57MB - put `orignal_images` directory in `dataset/` directory
* [torchvision](https://pytorch.org/) - `torchvision.models` contains the VGG16 and VGG19 model skeleton

### Dependecies
* [PyTorch](https://pytorch.org/)
* [opencv2](https://matplotlib.org/users/installing.html)
* [NumPy](https://www.scipy.org/install.html)
* [imutils](https://www.pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/)

```
python webcam.py -m model_path
```
**Options**
* `n` : next model
* `q`: quit
* `y`: capture image

## Files and Folder Structure
```
master_folder
 ~ dataset 
    ~ orignal_images
        *.jpg
        
 ~ images
    ~ out
        *.jpg
      *.jpg
    ~ capture
 ~ models
    *.pth
 ~ transforms
    *.pth
 ~ model_path
    *.pth
 *.py
```

## Todo!
* Web-app deployment of fast-neural-style (ONNX)

## Attribution
This implementation borrows markdown content, formatting and many implementation details from:
* Rusty Mina's [fast-neural-style in Torch](https://github.com/iamRusty/fast-neural-style-pytorch), and 
* the PyTorch Team's [PyTorch Examples: fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style)

