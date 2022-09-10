
## Project Title:

### Relative location of CT slices on axial axis (Dataset)

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
The modelling was done via two techniques: first, using an `nn.Embedding` layer with `nn.Linear` layers, and secondly, combining `nn.Embedding` layers, with `nn.LSTRM` layers and `nn.Linear` layers. Training was carried out for 100 epochs in each case, with an `Adam` optimizer and a learning rate of 1e-4.

The fully-connected neural network was built via the `PyTorch` library, for the regression task described in the `Overview` above. The network was wrapped via the `Skorch` API, to render it compatible with the `Scikit Learn` API. The final model was obtained after training for `20` epochs, with a learning rate of `1e-4`, and a batch size of `16`.

## Quick start
1. Navigate to the `scripts` folder:
```
$ cd scripts
```
2. Ensure compressed data file is decompressed into the `data` directory

3. Run the `main.py` file.
```
$ python3 main.py --arg_key arg_value
```
4. Arguments available include:
   ```
   - epochs
   - task ('classif' or 'regression')
   - lr (learning rate)
   - classes (None if 'regression', int if 'classif')
   - n_features (number of data features)
   - batch_size
   ```


## Performance
A curious phenomenon occured. Contrary to expectations, the Linear model performed better than the LSTM (~60% compared to ~58% test performance). This is startling. This is presently being looked into.

## Possible Improvements
Some attepmts may be made to improve the performance:
1. Play with the learning rate some more.
2. Try out other optimization schemes (presently using Adam).
3. Try out attention architectures and transformers.

## Other To-dos

1. Modularize the code.
