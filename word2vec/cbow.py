import torch
import torch.nn.functional as F
from math import sqrt
from collections import Counter
from data.utils import (
        read_text, 
        preprocess_text, 
        tokenize, 
        )
from ngram.implementation import (
        build_ngrams,
        build_ngram_counts_matrix,
        compute_ngram_average_log_likelihood,
        compute_ngram_nll,
        compute_ngram_perplexity
        )

def step_update(params,lr=0.1):
    for param in params:
        param.data-=lr*param.grad

def zero_grad(params):
    for param in params:
        param.grad=None

def main():
    text = read_text()
    processed_text = preprocess_text(text)
    tokens = tokenize(processed_text)
    num_tokens = len(tokens)
    V = sorted(set(tokens))
    word_to_idx = {word:idx for idx,word in enumerate(V)}
    idx_to_word = {idx:word for word,idx in word_to_idx.items()}
    N = 3
    embed_size = 30
    vocab_size = len(V)
    indexed_tokens = list(map(word_to_idx.get,tokens))
    ngrams = build_ngrams(tokens, n=N)
    idx_ngrams = build_ngrams(indexed_tokens, n=N)
    average_log_likelihood = compute_ngram_average_log_likelihood(ngrams)
    nll = compute_ngram_nll(average_log_likelihood)
    perplexity = compute_ngram_perplexity(average_log_likelihood)
    max_distance_to_target = 2 
    num_tokens_per_instance = 2*max_distance_to_target+1
    training_data = build_ngrams(indexed_tokens, n=num_tokens_per_instance)
    X = torch.tensor([instance[:max_distance_to_target]+instance[(max_distance_to_target+1):] for instance in training_data])
    y = torch.tensor([instance[max_distance_to_target] for instance in training_data])
    emb = torch.nn.EmbeddingBag(vocab_size,embed_size,mode="mean")
    # embed_mtx = (torch.randn(vocab_size,embed_size)/sqrt(vocab_size)).requires_grad_()
    # this isn't necessary but it makes the weight initialization explicit 
    # can be useful for pre-trained values
    with torch.no_grad():
        emb.weight.copy_(torch.randn(vocab_size, embed_size)/sqrt(embed_size))
    W = (torch.randn(embed_size,vocab_size)/sqrt(embed_size)).requires_grad_()
    b = torch.zeros(vocab_size,requires_grad=True)
    num_iter = 100000
    batch_size = 256
    for i in range(num_iter):
        ix = torch.randint(0,X.shape[0],(batch_size,))
        flat = X[ix].view(-1)
        offsets = torch.arange(0, flat.numel(), X.shape[1], device=flat.device)
        mean_embed = emb(flat, offsets)
        #mean_embed = embed_mtx[X[ix],:].mean(dim=1)
        logits = mean_embed@W+b
        loss = F.cross_entropy(logits,y[ix])
        if i%100==0:
            print(f"{loss=}")
        loss.backward()
        # step_update([embed_mtx,W,b],lr=0.05)
        # zero_grad([embed_mtx,W,b])
        step_update([emb.weight,W,b],lr=0.05)
        zero_grad([emb.weight,W,b])
