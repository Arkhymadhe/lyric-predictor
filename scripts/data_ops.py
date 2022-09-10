import os
import pandas as pd
import string
from torch.utils.data import Dataset, DataLoader

def clear_punctuation(text):
    text = ''.join([char for char in text if char not in all_punct])
    return text