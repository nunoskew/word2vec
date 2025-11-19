import torch
import torch.nn.functional as F
from math import sqrt
from collections import Counter
from data.utils import *
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

def read_analogy_validation_set(word_to_idx):
    val_tokens = read_analogy_test_set(filename='data/lotr-analogy-test-set.txt')
    processed_val_token_idxs = []
    processed_val_tokens = []
    for example in val_tokens:
        val_token_idxs = list(map(word_to_idx.get,example))
        if None not in val_token_idxs:
            processed_val_tokens.append(example)
            processed_val_token_idxs.append(val_token_idxs)
    return processed_val_tokens, processed_val_token_idxs

def topk_by_cosine(target_emb, embed_mtx, idx_to_word, k=10):
    if target_emb.dim() == 1:
        target_emb = target_emb.unsqueeze(0)
    sims = F.cosine_similarity(embed_mtx, target_emb, dim=1)
    top_vals, top_idxs = torch.topk(sims, k)
    return [(idx_to_word[i.item()], i.item(), v.item()) for v, i in zip(top_vals, top_idxs)]

def compute_analogy(example,embed_mtx):
    embed_example = embed_mtx[example]
    return embed_example[0]-embed_example[1]+embed_example[3]

def sig_loss(y_true,y_pred):
    return -torch.log(torch.sigmoid(y_pred[torch.arange(y_pred.shape[0]),y_true])).mean()

def noise_loss(logits,sampled_noise_idxs):
    noise_logits = logits.gather(dim=1, index=sampled_noise_idxs)
    return -torch.log(torch.sigmoid(-noise_logits)).sum(dim=1).mean()

def main():
    text = read_text()
    processed_text = preprocess_text(text)
    tokens = tokenize(processed_text)
    num_tokens = len(tokens)
    V = sorted(set(tokens))
    word_to_idx = {word:idx for idx,word in enumerate(V)}
    idx_to_word = {idx:word for word,idx in word_to_idx.items()}
    N = 3
    k = 10
    t = 1e-3
    embed_size = 300
    vocab_size = len(V)
    val_tokens, val_token_idxs = read_analogy_validation_set(word_to_idx)
    indexed_tokens = list(map(word_to_idx.get,tokens))
    freqs = Counter(indexed_tokens)
    freqs = torch.tensor([freqs[i] for i in range(vocab_size)],dtype=float)
    unigram_probs = freqs/freqs.sum()
    p = (1-torch.sqrt(t/unigram_probs)).clip(0)
    mask = ~p[indexed_tokens].bernoulli().bool()
    indexed_tokens = torch.tensor(indexed_tokens)
    masked_tokens = torch.tensor(indexed_tokens[mask])
    max_distance_to_target = 5
    num_tokens_per_instance = 2*max_distance_to_target+1
    training_data = build_ngrams(masked_tokens, n=num_tokens_per_instance)
    X = torch.tensor([instance[max_distance_to_target] for instance in training_data])
    y = torch.tensor([instance[:max_distance_to_target]+instance[(max_distance_to_target+1):] for instance in training_data])
    embed_mtx = (torch.randn(vocab_size,embed_size)/sqrt(vocab_size)).requires_grad_()
    W = (torch.randn(embed_size,vocab_size)/sqrt(embed_size)).requires_grad_()
    b = torch.zeros(vocab_size,requires_grad=True)
    num_iter = 100000
    batch_size = 12
    num_epochs = 200
    expanded_training_data = torch.vmap(torch.cartesian_prod)(X.view(-1,1),y)
    idxs = torch.arange(-max_distance_to_target,max_distance_to_target+1)
    mid = len(idxs)//2
    idxs = torch.concat([idxs[:mid],idxs[mid+1:]])
    for epoch in range(num_epochs):
        R = torch.randint(1,max_distance_to_target+1,(expanded_training_data.shape[0],))
        bool_idxs = torch.vmap(lambda r:idxs.abs()<=r)(R)
        sampled_expanded_training_data = expanded_training_data[bool_idxs]
        X_expanded, y_expanded = sampled_expanded_training_data[:,0],sampled_expanded_training_data[:,1]
        for i in range(X_expanded.shape[0]//batch_size):
            ix = torch.randint(0,X_expanded.shape[0],(batch_size,))
            logits = embed_mtx[X_expanded[ix]]@W+b
            sampled_noise = unigram_probs.multinomial(num_samples=(batch_size*k),replacement=True).view(batch_size,k)
            loss = sig_loss(y_true=y_expanded[ix],y_pred=logits)
            nl = noise_loss(logits=logits,sampled_noise_idxs=sampled_noise)
            loss += nl
            if i%1000==0:
                for example in val_token_idxs:
                    analogy_output = compute_analogy(example, embed_mtx)
                    expected_output = idx_to_word[example[2]]
                    closest = topk_by_cosine(analogy_output, embed_mtx, idx_to_word, k=5)
                    if closest[0][0]==expected_output:
                        print(f'WOW we got one right!! {example=}')
                    else:
                        print(f"expected {expected_output}, got {closest[0][0]} in {list(map(idx_to_word.get,example))}. {closest=}")
                print(f"{loss=}")
            loss.backward()
            step_update([embed_mtx,W,b],lr=0.001)
            zero_grad([embed_mtx,W,b])
