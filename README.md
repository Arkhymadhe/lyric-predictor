
## Project Title:

### Lyric Genre Predictor

---

## Stack
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-360/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://www.jupyter.org)
[![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)](https://www.jetbrains.com/pycharm/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)](http://git-scm.com/)
[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg?&style=for-the-badge)](https://saythanks.io/to/kennethreitz)

## Overview

The dataset comprises text and metadata. The text is a sequence of lyrics from the song, and the metadata is the musci genre the song belongs to (rap or non-rap). As such, this is a binary classificication NLP problem (many-to-one).

## Training procedure
The modelling was done via two techniques: first, using an `nn.Embedding` layer with `nn.Linear` layers, and secondly, combining `nn.Embedding` layers, with `nn.LSTRM` layers and `nn.Linear` layers. Training was carried out for 100 epochs in each case, with an `Adam` optimizer and a learning rate of `1e-4` and a batch size of `32`.

## Performance
As expected, the LSTM model outperformed the linear model (`~84%` as compared to `~60%` test performance). However, the LSTM is prone to overfitting. This will have to be addressed.

## Possible Improvements
Some attepmts may be made to improve the performance:
1. Play with the learning rate some more.
2. Try out other optimization schemes (presently using Adam).
3. Try out attention architectures and transformers.
4. Tackle overfitting with the LSTM.

## Other To-dos

1. Modularize the code.
