import re
import torch
from . import base_dir
from collections import Counter
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
from math import log2,log,exp
def read_text(filename="lotr-the-fellowship-of-the-ring.txt"):
    return open(base_dir/filename,encoding="utf-8").read()

def save_processed(text,filename="clean-lotr-the-fellowship-of-the-ring.txt"):
    with open(base_dir/filename, "w", encoding="utf-8") as f:
        f.write(text)

def preprocess_text(text, save=False):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    if save:
        save_processed(text,filename=f"clean-lotr-the-fellowship-of-the-ring-{datetime.now().strftime('%d%b%Y_%HH%M').upper()}.txt")
    return text

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

