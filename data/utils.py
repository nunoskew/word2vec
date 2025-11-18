import re
import torch
import spacy
from . import base_dir
from collections import Counter
from datasets import load_dataset
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
from math import log2,log,exp



def read_google_analogy_test_set(filename="data/google-analogy-test-set.txt"):
    with open(filename, "rb") as f:
        google_analogy_dataset = [line.decode("utf-8").strip() for line in f.readlines()[1:]]
    google_analogy_dataset = [analogy.lower().split(' ') for analogy in google_analogy_dataset]
    return google_analogy_dataset

def read_text(filename="lotr-the-fellowship-of-the-ring.txt"):
    return open(base_dir/filename,encoding="utf-8").read()

def save_processed(text,filename="clean-lotr-the-fellowship-of-the-ring.txt"):
    with open(base_dir/filename, "w", encoding="utf-8") as f:
        f.write(text)

def preprocess_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r"[^a-z']+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def pipeline():
    google_analogy_dataset = read_google_analogy_test_set()
    analogy_words = {w for row in google_analogy_dataset for w in row}
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    raw_corpus = "\n".join(ds["train"]['text'])
    processed_text = preprocess_text(raw_corpus)
    tokens = tokenize(processed_text)
    freqs = Counter(tokens)
    min_count = 200
    V = {w for w, c in freqs.items() if c >= min_count or w in analogy_words}
    special_tokens = {"<unk>"}
    V = special_tokens | V 
    num_tokens = len(tokens)
    vocab_size = len(V)
    word_to_idx = {word:idx for idx,word in enumerate(V)}
    idx_to_word = {idx:word for word,idx in word_to_idx.items()}
    indexed_tokens = list(map(lambda x: word_to_idx.get(x,word_to_idx['<unk>']),tokens))

