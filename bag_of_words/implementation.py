from typing import List
from doctest import testmod
from collections import Counter

def bow(V,ngrams: List[tuple]) -> List[dict]:
    """
    Bag of words representation of a list of ngrams.
    >>> from collections import Counter
    >>> V = ['the','person','cat','sat']
    >>> trigrams = [('the','cat','sat'),('the','person','sat')]
    >>> expected_counts = [{'the':1,'person':0,'cat':1,'sat':1},
    ...                    {'the':1,'person':1,'cat':0,'sat':1}]
    >>> bow(V,trigrams)==expected_counts
    True
    """
    baseline_counts = {word:0 for word in V}
    counts = []
    for ngram in ngrams:
        baseline_counts.update(Counter(ngram))
        counts.append(baseline_counts)
        baseline_counts = {word:0 for word in V}
    return counts

if __name__=="__main__":
    testmod()






def main():
    text = read_text()
    processed_text = preprocess_text(text)
    tokens = tokenize(processed_text)
    unigrams = Counter(tokens)
    V = sorted(set(tokens))
    word_to_idx = {word:idx for idx,word in enumerate(V)}
    idx_to_word = {idx:word for word,idx in word_to_idx.items()}
    N = 3
    indexed_tokens = list(map(word_to_idx.get,tokens))
    ngrams = build_ngrams(tokens, n=N)
    idx_ngrams = build_ngrams(indexed_tokens, n=N)
    average_log_likelihood = compute_ngram_average_log_likelihood(ngrams)
    nll = compute_ngram_nll(average_log_likelihood)
    perplexity = compute_ngram_perplexity(average_log_likelihood)
        
