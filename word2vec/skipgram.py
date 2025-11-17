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

def hierarchical_softmax_loss(H, targets):
    """
    this is too hard for me right now.

    path_nodes[w] : list[int] internal node ids on th epath
    path_bits[w]: list[int] 0/1 decisions aligned with those nodes
    """
    total = H.new_zeros(())
    for i in range(H.shape[0]):
        w = int(targets[i])
        nodes = torch.tensor(path_nodes[w], device=H.device)
        bits = torch.tensor(path_bits[w], device=H.device, dtype=H.dtype)
        s = (U[nodes] @ H[i])
        total+= F.binary_cross_entropy_with_logits(s,bits,reduction='sum')
    return total / max(1, H.shape[0])

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
    max_distance_to_target = 5
    num_tokens_per_instance = 2*max_distance_to_target+1
    training_data = build_ngrams(indexed_tokens, n=num_tokens_per_instance)
    X = torch.tensor([instance[max_distance_to_target] for instance in training_data])
    y = torch.tensor([instance[:max_distance_to_target]+instance[(max_distance_to_target+1):] for instance in training_data])
    embed_mtx = (torch.randn(vocab_size,embed_size)/sqrt(vocab_size)).requires_grad_()
    W = (torch.randn(embed_size,vocab_size)/sqrt(embed_size)).requires_grad_()
    b = torch.zeros(vocab_size,requires_grad=True)
    num_iter = 100000
    batch_size = 512
    num_epochs = 2
    expanded_training_data = torch.vmap(torch.cartesian_prod)(X.view(-1,1),y)
    idxs = torch.arange(1,max_distance_to_target+1)
    idxs = torch.cat([torch.flip(idxs*-1,dims=(0,)),idxs])
    for epoch in range(num_epochs):
        R = torch.randint(1,max_distance_to_target+1,(expanded_training_data.shape[0],))
        bool_idxs = torch.vmap(lambda r:idxs.abs()<=r)(R)
        sampled_expanded_training_data = expanded_training_data[bool_idxs]
        X_expanded, y_expanded = sampled_expanded_training_data[:,0],sampled_expanded_training_data[:,1]
        for i in range(X_expanded.shape[0]//batch_size):
            ix = torch.randint(0,X_expanded.shape[0],(batch_size,))
            logits = embed_mtx[X_expanded[ix]]@W+b
            loss = F.cross_entropy(logits,y_expanded[ix])
            if i%100==0:
                print(f"{loss=}")
            loss.backward()
            step_update([embed_mtx,W,b],lr=0.02)
            zero_grad([embed_mtx,W,b])
