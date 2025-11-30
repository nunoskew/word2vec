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
    return (x+eps)/(x.norm(dim=1,keepdim=True)+eps)


def compute_analogy(embed_examples):
    return embed_examples[0]-embed_examples[1]+embed_examples[3]

def sig_loss(y_true,y_pred):
    return -torch.log(torch.sigmoid(y_pred[torch.arange(y_pred.shape[0]),y_true])).mean()

def noise_loss(logits,sampled_noise_idxs):
    noise_logits = logits.gather(dim=1, index=sampled_noise_idxs)
    return -torch.log(torch.sigmoid(-noise_logits)).sum(dim=1).mean()

def topk_by_cosine(vec, embed_mtx, idx_to_word, k=10):
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)  # [1, D]

    vec_norm = F.normalize(vec, p=2, dim=-1)           # [1, D]
    embed_norm = F.normalize(embed_mtx, p=2, dim=1)    # [V, D]

    scores = vec_norm @ embed_norm.T                  # [1, V]
    scores = scores.squeeze(0)                        # [V]

    values, indices = torch.topk(scores, k)
    values = values.detach().cpu()
    indices = indices.detach().cpu()

    return [(idx_to_word[int(i)], float(values[j])) for j, i in enumerate(indices)]

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu") 
    text = read_text('the_hobbit.txt')
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
    V_in  = torch.randn(vocab_size, embed_size, device=device) / sqrt(vocab_size)
    V_out = torch.randn(vocab_size, embed_size, device=device) / sqrt(vocab_size)

    V_in.requires_grad_()
    V_out.requires_grad_()
    num_iter = 100000
    batch_size = 4096*2
    num_epochs = 4000
    expanded_training_data = torch.vmap(torch.cartesian_prod)(X.view(-1,1),y).to(device)
    idxs = torch.arange(-max_distance_to_target,max_distance_to_target+1,device=device)
    mid = len(idxs)//2
    idxs = torch.concat([idxs[:mid],idxs[mid+1:]])
    optimizer = torch.optim.Adam([V_in,V_out], lr=0.001)
    for epoch in range(num_epochs):
        R = torch.randint(1,max_distance_to_target+1,(expanded_training_data.shape[0],),device=device)
        bool_idxs = torch.vmap(lambda r:idxs.abs()<=r)(R)
        sampled_expanded_training_data = expanded_training_data[bool_idxs]
        X_expanded, y_expanded = sampled_expanded_training_data[:,0],sampled_expanded_training_data[:,1]
        for i in range(X_expanded.shape[0]//batch_size):
            ix = torch.randint(0,X_expanded.shape[0],(batch_size,),device=device)
            centers   = X_expanded[ix]        # [B]
            targets   = y_expanded[ix]        # [B]
            sampled_noise = unigram_probs.multinomial(num_samples=(batch_size*k),replacement=True).view(batch_size,k).to(device)
            negatives = sampled_noise         # [B, K]

            v_c = V_in[centers]               # [B, d]
            u_o = V_out[targets]              # [B, d]
            u_k = V_out[negatives]            # [B, K, d]

            pos_scores = (v_c * u_o).sum(-1)            # [B]
            neg_scores = (v_c.unsqueeze(1) * u_k).sum(-1)  # [B, K]
            loss_pos = -torch.log(torch.sigmoid(pos_scores)).mean()
            loss_neg = -torch.log(torch.sigmoid(-neg_scores)).sum(1).mean()
            loss = loss_pos + loss_neg
            if i%1000==0:
                for example in val_token_idxs:
                    analogy_embed_examples = V_in[example]
                    analogy_output = compute_analogy(analogy_embed_examples)
                    expected_output = idx_to_word[example[2]]
                    closest = topk_by_cosine(analogy_output, V_in, idx_to_word, k=10)
                    if closest[0][0]==expected_output:
                        print(f'WOW we got one right!! {example=}')
                    else:
                        print(f"expected {expected_output}, got {closest[0][0]} in {list(map(idx_to_word.get,example))}. {closest=}")
                print(f"{loss=}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=="__main__":
    main()
