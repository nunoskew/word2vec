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
import nltk
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))

def step_update(params,lr=0.1):
    for param in params:
        param.data-=lr*param.grad

def zero_grad(params):
    for param in params:
        param.grad=None

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

def normalize(x):
    eps = 1e-5
    return (x+eps)/(x.norm(dim=-1,keepdim=True)+eps)

def topk_by_cosine(vec, embed_mtx, idx_to_word, k=10):
    scores = vec @ embed_mtx.T  
    values, indices = torch.topk(scores, k)
    values = values.detach().cpu()
    indices = indices.detach().cpu()
    return [(idx_to_word[int(i)], float(values[j]))
            for j, i in enumerate(indices)]

def compute_analogy(example,embed_mtx):
    embed_example = embed_mtx[example]
    return embed_example[0]-embed_example[1]+embed_example[3]

def sig_loss(y_true,y_pred):
    return -torch.log(torch.sigmoid(y_pred[torch.arange(y_pred.shape[0]),y_true])).mean()

def noise_loss(logits,sampled_noise_idxs):
    noise_logits = logits.gather(dim=1, index=sampled_noise_idxs)
    return -torch.log(torch.sigmoid(-noise_logits)).sum(dim=1).mean()

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu") 
    text = read_text()
    processed_text = preprocess_text(text)
    tokens = tokenize(processed_text)
    tokens = [token for token in tokens if token not in stop_words]
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
    X = torch.tensor([instance[max_distance_to_target] for instance in training_data],device=device)
    y = torch.tensor([instance[:max_distance_to_target]+instance[(max_distance_to_target+1):] for instance in training_data],device=device)
    embed_mtx = (torch.randn(vocab_size,embed_size,device=device))/sqrt(vocab_size)
    embed_mtx.requires_grad_()
    W = (torch.randn(embed_size,vocab_size,device=device))/sqrt(embed_size)
    W.requires_grad_()
    b = torch.zeros(vocab_size,device=device,requires_grad=True)
    num_iter = 100000
    batch_size = 4096
    num_epochs = 2000
    expanded_training_data = torch.vmap(torch.cartesian_prod)(X.view(-1,1),y).to(device)
    idxs = torch.arange(-max_distance_to_target,max_distance_to_target+1,device=device)
    mid = len(idxs)//2
    idxs = torch.concat([idxs[:mid],idxs[mid+1:]])
    optimizer = torch.optim.Adam([embed_mtx,W,b], lr=0.001)
    for epoch in range(num_epochs):
        R = torch.randint(1,max_distance_to_target+1,(expanded_training_data.shape[0],),device=device)
        bool_idxs = torch.vmap(lambda r:idxs.abs()<=r)(R)
        sampled_expanded_training_data = expanded_training_data[bool_idxs]
        X_expanded, y_expanded = sampled_expanded_training_data[:,0],sampled_expanded_training_data[:,1]
        for i in range(X_expanded.shape[0]//batch_size):
            ix = torch.randint(0,X_expanded.shape[0],(batch_size,),device=device)
            logits = embed_mtx[X_expanded[ix]]@W+b
            sampled_noise = unigram_probs.multinomial(num_samples=(batch_size*k),replacement=True).view(batch_size,k).to(device)
            loss = sig_loss(y_true=y_expanded[ix],y_pred=logits)
            nl = noise_loss(logits=logits,sampled_noise_idxs=sampled_noise)
            loss += nl
            if i%1000==0:
                for example in val_token_idxs:
                    analogy_output = compute_analogy(example, embed_mtx)
                    expected_output = idx_to_word[example[2]]
                    closest = topk_by_cosine(normalize(analogy_output), normalize(embed_mtx), idx_to_word, k=5)
                    if closest[0][0]==expected_output:
                        print(f'WOW we got one right!! {example=}')
                    else:
                        print(f"expected {expected_output}, got {closest[0][0]} in {list(map(idx_to_word.get,example))}. {closest=}")
                print(f"{loss=}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss.backward()
            # step_update([embed_mtx,W,b],lr=0.01)
            # zero_grad([embed_mtx,W,b])

if __name__=="__main__":
    main()
