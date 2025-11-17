from data.utils import *
from typing import List
from functools import reduce
from torch.nn.functional import softmax

def compute_average_log_likelihood(P: torch.Tensor,observed_pairs: List[Tuple[int]]|None=None):
    """
    Compute the average log-likelihood of a conditional distribution matrix.

    Interprets `P` as a conditional probability table where `P[i, j] = p(next=j | prev=i)`.
    If `observed_pairs` is provided, the average is computed only over those (prev, next)
    index pairs. Otherwise the average is over **all** entries in `P`.

    Args:
        P (torch.Tensor): A 2-D tensor of shape (V, V) with row-wise probabilities
            (each row sums to 1). `V` is the vocabulary size.
        observed_pairs (list[tuple[int, int]] | None): Optional list of `(prev_idx, next_idx)`
            pairs drawn from the observed data. If `None`, compute over the full matrix.

    Returns:
        torch.Tensor: Scalar tensor, the mean of `log P[rows, cols]` over the chosen entries.

    Raises:
        AssertionError: If `P` is not 2-D, or contains invalid probabilities.

    Example:
        >>> import torch
        >>> from math import isclose, log
        >>> P = torch.tensor([[0.6, 0.4],
        ...                   [0.1, 0.9]], dtype=torch.float32)
        >>> pairs = [(0, 0), (0, 1), (1, 1)]
        >>> actual = compute_average_log_likelihood(P, pairs)
        >>> expected = (1/3)*(log(0.6)+log(0.4)+log(0.9))
        >>> isclose(actual.item(),expected,rel_tol=0,abs_tol=1e-5)
        True
    """
    expected_num_dims = 2
    assert P.ndim==expected_num_dims, f"Wrong number of dimensions. Expected {expected_num_dims}, got {P.ndim}."
    def _build_average_ll(P):
        if observed_pairs:
            rows,cols = zip(*observed_pairs)
            return P[rows,cols].log().mean()
        return P.log().mean() 
    out = _build_average_ll(P)
    assert out<0, f"Average Log Likelihood is negative: {out}"
    return out

def compute_nll(average_ll):
    return -average_ll

def cross_entropy(y_true: torch.Tensor,y_pred: torch.Tensor):
    """
    Compute cross-entropy between a target distribution and predicted probabilities.

    This treats each row as a categorical distribution. Typical use:
    - `y_true` is either:
        (a) one-hot rows of shape (N, V), **or**
        (b) index vector of shape (N,) with class indices in [0, V).
      (If you only support one-hot, document it here.)
    - `y_pred` contains probabilities (rows sum to 1).

    Args:
        y_true (torch.Tensor): Ground-truth labels; shape (N, V) for one-hot or (N,) for indices.
        y_pred (torch.Tensor): Predicted probabilities; shape (N, V). Rows must sum to 1.

    Returns:
        torch.Tensor: Scalar tensor with mean cross-entropy across the batch.

    Raises:
        AssertionError: If shapes are incompatible or rows of `y_pred` do not sum to 1.

    Example (one-hot target):
        >>> import torch
        >>> y_true = torch.tensor([0,1])
        >>> y_pred = torch.tensor([[0.8, 0.2],
        ...                        [0.1, 0.9]])
        >>> actual = cross_entropy(y_true, y_pred)
        >>> torch.allclose(actual,torch.tensor([-log(0.8), -log(0.9)]).mean())
        True
    """
    eps = 1e-2
    row_sums = y_pred.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), f"Predictions do not sum up to 1."
    batch = torch.arange(y_pred.size(0), device=y_pred.device)
    out = -y_pred[batch,y_true].log().mean()
    assert out>0, f"Cross Entropy is always positive: {out}"
    return out

def compute_perplexity(average_ll: float):
    assert average_ll<0, f"Average log likelihood is negative, got {out}"
    out=(-average_ll).exp() 
    assert out>0, f"Perplexity is non-positive, got {out}"
    return out

def g(C: torch.Tensor, W:torch.Tensor, b:torch.Tensor, embedding_size:int, vocab_size: int) -> torch.Tensor:
    """
    Compute logits for next-word prediction from an embedding lookup and affine layer.

    Conceptually:
      1) `C` stores word embeddings; shape (V, D) where D = `embedding_size`.
      2) The model maps a context representation (built elsewhere) through:
             logits = context @ W + b
         where `W` has shape (D, V) or (context_dim, V), depending on your design.

    Args:
        C (torch.Tensor): Embedding matrix of shape (V, D).
        W (torch.Tensor): Weight matrix mapping from hidden/context to vocabulary; shape (D, V) or compatible.
        b (torch.Tensor): Bias vector; shape (V,).
        embedding_size (int): D, the embedding dimensionality.
        vocab_size (int): V, the vocabulary size.

    Returns:
        torch.Tensor: Logits of shape (V,) or (N, V), depending on your context input usage.

    Notes:
        This function name is from the Bengio et al. (2003) notation where `g` often denotes
        the final scoring function before softmax.
    """
    eps = 1e-1
    expected_C_ndim = 2
    assert C.ndim==expected_C_ndim, f"Input has wrong number of dimensions. Expected {expected_C_ndim}, got {C.ndim}"
    assert C.shape[1]==embedding_size, f"Input has wrong embedding size. Expected {embedding_size}, got {C.shape[1]}"
    def forward(C,W,b):
        return softmax(C@W+b,dim=1)
    out = forward(C,W,b)
    expected_out_ndim = 2
    assert out.ndim==expected_out_ndim, f"Input has wrong number of dimensions. Expected {expected_out_ndim}, got {out.ndim}"
 
    assert out.shape[1]==vocab_size, f"Output has wrong number of columns. Expected {vocab_size}, got {out.shape[1]}"
    assert all((out.sum(axis=1)-torch.ones(out.shape[0]))<eps), f"Output does not sum up to 1."
    return out


def zero_grad(params):
    for p in params:
        p.grad=None

def step_update(params):
    for p in params:
        p.data+= -0.1*p.grad

def sample_from_probability_distribution(P: torch.Tensor, word_to_idx: dict[str,int], idx_to_word: dict[int,str], starting_word:str='.') -> None:
    """
    Sample a word sequence from a row-stochastic probability matrix `P`.

    Assumes `P[i, j] = p(next=j | prev=i)`. Sampling begins at `starting_word`
    (default: '.'), then repeatedly samples the next index according to the
    corresponding row of `P`.

    Args:
        P (torch.Tensor): Shape (V, V). Each row sums to 1.
        word_to_idx (dict[str, int]): Mapping from token string to index.
        idx_to_word (dict[int, str]): Reverse mapping.
        starting_word (str): Token to start from (should exist in `word_to_idx`).
    
    Returns:
        None, just prints the samples.
    """
    last_idx = word_to_idx[starting_word] 
    sentence_idxs = []
    for _ in range(100):
        idx = torch.multinomial(P[last_idx], num_samples=1, replacement=True).item()
        sentence_idxs.append(idx_to_word[idx])
        last_idx = idx
    print(' '.join(sentence_idxs))

def build_P(ngram_counts_matrix):
    """
    Convert an n-gram count matrix to a row-stochastic probability matrix.

    Adds a small `eps` for numerical stability and normalizes each row so that
    it sums to 1. Typically `ngram_counts_matrix[i, j]` stores how many times
    the bigram `(i, j)` occurs in the corpus.

    Args:
        ngram_counts_matrix (torch.Tensor): 2-D counts of shape (V, V).

    Returns:
        torch.Tensor: Probability matrix `P` with the same shape, where each row sums to 1.

    Raises:
        AssertionError: If input is not 2-D or is not square.

    Example:
        >>> import torch
        >>> counts = torch.tensor([[2., 1.],
        ...                        [0., 3.]])
        >>> P = build_P(counts)
        >>> expected = torch.tensor([[2/3,1/3],[0.0,1.0]])
        >>> P.shape
        torch.Size([2, 2])
        >>> torch.allclose(P, expected, atol=1e-5)
        True
    """
    eps = 1e-5
    expected_num_dims_ngram_mtx = 2
    assert ngram_counts_matrix.ndim==expected_num_dims_ngram_mtx, f"Wrong number of dimension. Expected {expected_num_dims_ngram_mtx}, got {ngram_counts_matrix.ndim}" 
    assert ngram_counts_matrix.shape[0]==ngram_counts_matrix.shape[1], f"Matrix is not square"
    def _build_P(ngram_counts_matrix):
        # question: why can't i normalize by the sum of P?
        return (ngram_counts_matrix+eps)/(ngram_counts_matrix+eps).sum(dim=1,keepdim=True)
    out = _build_P(ngram_counts_matrix)
    expected_P_row_sums = 1
    assert all((out.sum(dim=1)-expected_P_row_sums).abs()<eps), f"Conditional Probability distribution rows do not sum up to {expected_P_row_sums}. Max diff was {out.sum(dim=1).max()}"
    return out

if __name__=="__main__":
    from doctest import testmod
    testmod()
