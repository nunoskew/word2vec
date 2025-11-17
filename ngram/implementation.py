import torch
from collections import Counter
from typing import List, Tuple
from math import log,exp

def build_ngrams(tokens: List[str]|List[int], n=2) -> List[Tuple[str]]|List[Tuple[int]]:
    """
    Build n-grams from a sequence of tokens.

    Consecutive groups of *n* tokens are returned as tuples.  
    For example, with n=2 this returns bigrams, with n=3 trigrams.

    Example:
        >>> build_ngrams(["the", "cat", "sat", "down"], n=2)
        [('the', 'cat'), ('cat', 'sat'), ('sat', 'down')]
        >>> build_ngrams([1, 2, 3, 4], n=3)
        [(1, 2, 3), (2, 3, 4)]

    Returns:
        list[tuple]: a list of n-length tuples of consecutive tokens.
    """
    shifted_tokens = (tokens[i:] for i in range(n))
    return list(zip(*shifted_tokens))

def build_ngram_counts_matrix(ngrams: List[Tuple[int]]):
    """
    Build a bigram count matrix from a list of (word_index₁, word_index₂) pairs.

    Each entry (i, j) in the returned matrix contains the number of times
    token i is followed by token j in the given list of bigrams.

    Example:
        >>> ngrams = [(0, 1), (0, 2), (1, 2), (2, 0), (0, 1)]
        >>> build_ngram_counts_matrix(ngrams)
        tensor([[0., 2., 1.],
                [0., 0., 1.],
                [1., 0., 0.]])

        # Explanation:
        # - 0→1 occurs twice
        # - 0→2 occurs once
        # - 1→2 occurs once
        # - 2→0 occurs once

    Returns:
        torch.Tensor: a square matrix of shape (V, V),
        where V is the vocabulary size inferred from the indices.
    """
    assert isinstance(ngrams[0],tuple)
    assert len(ngrams[0])==2, f"Function just works for bigrams, the input has examples such as {ngrams[0]}"
    assert isinstance(ngrams[0][0],int), f"Tuple entries should be indexes, got this type of tuple {ngrams[0]}"
    vocab_size = len(set([x for a,b in set(ngrams) for x in [a,b]]))
    counts = torch.zeros((vocab_size,vocab_size))
    for w1_idx,w2_idx in ngrams:
        counts[w1_idx,w2_idx] += 1
    return counts

def count_ngrams(ngrams: List[Tuple[int]]) -> dict[Tuple[int],float]:
    return Counter(ngrams)

def compute_ngram_normalized_counts(ngrams: List[Tuple[int]]) -> dict[Tuple[int],float]:
    """
    Compute the empirical conditional probabilities P(w|h) for each observed n-gram.

    Each n-gram (h, w) is assigned:
        P(w|h) = count(h, w) / count(h)

    Example:
        >>> ngrams = [(0, 1), (0, 2), (1, 2), (2, 0), (0, 1)]
        >>> out = compute_ngram_normalized_counts(ngrams)
        >>> round(out[(0, 1)], 2)
        0.67
        >>> round(out[(0, 2)], 2)
        0.33
        >>> round(out[(1, 2)], 2)
        1.0
        >>> round(out[(2, 0)], 2)
        1.0

    Returns:
        dict[tuple[int], float]: mapping each n-gram (h, w)
        to its conditional probability P(w|h).
    """
    counts = Counter(ngrams)
    context = [ngram[:-1] for ngram in ngrams]
    context_counts = Counter(context)
    return {ngram:counts/context_counts[ngram[:-1]] for ngram,counts in counts.items()}

def compute_ngram_log_normalized_counts(ngrams: List[Tuple[int]]) -> dict[Tuple[int],float]:
    """
    Compute the empirical log conditional probabilities log P(w|h) for each observed n-gram.

    Each n-gram (h, w) is assigned:
        log P(w|h) = log( count(h, w) / count(h) )

    Example:
        >>> from math import log, isclose
        >>> ngrams = [(1, 2), (1, 3), (2, 1)]
        >>> out = compute_ngram_log_normalized_counts(ngrams)
        >>> set(out.keys()) == {(1, 2), (1, 3), (2, 1)}
        True
        >>> isclose(out[(1, 2)], log(1/2))
        True
        >>> isclose(out[(1, 3)], log(1/2))
        True
        >>> isclose(out[(2, 1)], log(1))
        True

    Returns:
        dict[tuple, float]: log probabilities for each observed (history, next) pair.
    """
    counts = Counter(ngrams)
    context_counts = Counter([ngram[:-1] for ngram in ngrams])
    return {ngram:log(counts/context_counts[ngram[:-1]]) for ngram,counts in counts.items()}


def compute_ngram_average_log_likelihood(ngrams):
    """
    Compute the empirical average log-likelihood per observed n-gram:
        (1/T) * sum_{(h,w)} c(h,w) * log( c(h,w) / c(h) )

    Example:
        Suppose our data has two histories:
            - Context 1 appears twice: 1→2, 1→3
            - Context 2 appears once: 2→1

        So:
            p(2|1) = 1/2
            p(3|1) = 1/2
            p(1|2) = 1

        >>> ngrams = [(1,2), (1,3), (2,1)]
        >>> round(compute_ngram_average_log_likelihood(ngrams), 6)
        -0.462098

        (Because: (1/3)*( log(1/2)+log(1/2)+log(1) ) = -0.462098)

    """
    counts = Counter(ngrams)
    num_ngrams = len(ngrams)
    context_counts = Counter([ngram[:-1] for ngram in ngrams])
    log_norm_counts = {ngram:log(ngram_count/context_counts[ngram[:-1]]) for ngram,ngram_count in counts.items()}
    return sum([counts[ngram]*log_norm_count for ngram,log_norm_count in log_norm_counts.items()])/num_ngrams

def compute_ngram_nll(average_log_likelihood):
    return -average_log_likelihood

def compute_ngram_perplexity(average_log_likelihood):
    return exp(-average_log_likelihood)


if __name__=="__main__":
    from doctest import testmod
    testmod()
