import re
import torch
from . import base_dir
from collections import Counter
from datasets import load_dataset
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
from math import log2,log,exp



def read_analogy_test_set(filename="data/google-analogy-test-set.txt"):
    with open(filename, "rb") as f:
        analogy_dataset = [line.decode("utf-8").strip() for line in f.readlines()[1:]]
    analogy_dataset = [analogy.lower().split(' ') for analogy in analogy_dataset]
    return analogy_dataset

def read_text(filename="lotr-the-fellowship-of-the-ring.txt"):
    return open(base_dir/filename,encoding="utf-8").read()

def save_processed(text,filename="clean-lotr-the-fellowship-of-the-ring.txt"):
    with open(base_dir/filename, "w", encoding="utf-8") as f:
        f.write(text)

def preprocess_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'\d+', '<num>', text)
    return text

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

