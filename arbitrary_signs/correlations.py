"""correlations"""

import itertools as it
from Levenshtein import distance as levenshtein
import logging
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine
import tqdm
from urllib.request import urlopen

from arbitrary_signs import url_base, valid_language_exts
from arbitrary_signs.utils import make_safe_dir

logging.getLogger().setLevel(logging.INFO)


def create_url(ext, base):
    if ext not in valid_language_exts:
        raise Exception("{} is bad language extension.")
    return base.format(ext)


def download_fasttext_data(url, verbose=True):
    """Returns tuple of (dict, int, int)"""
    if verbose:
        logging.info("Downloading data from {}".format(url))

    data = {}
    f = urlopen(url)
    header = f.readline()
    header_data = header.split()
    vocab_size, hidden_size = int(header_data[0]), int(header_data[1])
    pbar = tqdm.tqdm(total=vocab_size)
    for line_idx, line in tqdm.tqdm(enumerate(f.readlines())):
        elements = line.split()
        word = elements[0]
        vec = np.array(list(map(float, elements[1:])))
        data[word] = vec
        pbar.update()
    pbar.close()
    return data, vocab_size, hidden_size


def get_vocab_sample(model, n=500):
    """

    Parameters
    ----------
    model: dict
        Keys are words, values are vector repr.
    n: int [Default: 500]
        Number of words to sample.

    Returns
    -------
    tuple (np.array, np.array, np.array)
        Vector, words, random ordered words.

    """
    vocab_words_1 = np.array(list(model.keys()))
    vocab_words_2 = np.array(list(model.keys()))
    vector_repr = np.array(list(model.values()))

    idxs = np.random.choice(range(len(vocab_words_1)), size=n)
    words_sample = vocab_words_1[idxs]
    words_sample_random = vocab_words_2[idxs]
    vector_sample = vector_repr[idxs]
    np.random.shuffle(words_sample_random)
    # Basic check on randomization
    assert words_sample[0] != words_sample_random[0]
    return vector_sample, words_sample, words_sample_random


def get_pairwise_distances(items, dist_fn=lambda x: x):
    pairs = list(it.product(items, items))
    final_pairs = []
    dists = []
    for a, b in pairs:
        dist = dist_fn(a, b)
        # Note (BP): You may need to complicate this...
        if dist == 0:
            continue
        dists.append(dist)
        final_pairs.append((a, b))
    return dists, final_pairs


def get_distance_correlations(model, n=500):
    vals, words, random_words = get_vocab_sample(model, n)
    val_dists, _ = get_pairwise_distances(vals, cosine)
    word_dists, _ = get_pairwise_distances(words, levenshtein)
    random_word_dists, _ = get_pairwise_distances(random_words, levenshtein)
    return np.corrcoef(val_dists, word_dists)[0][1], \
           np.corrcoef(val_dists, random_word_dists)[0][1]


def run_distance_correlations(model, n_sims=100, n_sample=100):
    real_corrs = []
    random_corrs = []
    for _ in tqdm.tqdm(range(n_sims)):
        real_corr, random_corr = get_distance_correlations(model, n_sample)
        real_corrs.append(real_corr)
        random_corrs.append(random_corr)
    return np.array(real_corrs), np.array(random_corrs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language-ext", type=str, default="en",
                        help="Language extension [Default: 'en'].")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples [Default: 100].")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Sample size [Default: 500].")
    parser.add_argument("--cache-model", action='store_true', default=True,
                        help="Pickle model file [Default: True].")
    parser.add_argument("--out-dir", type=str, default="./outputs",
                        help="Output directory [Default: ./outputs].")

    args = parser.parse_args()

    language_ext = args.language_ext
    n_samples = args.n_samples
    sample_size = args.sample_size

    make_safe_dir(args.out_dir)

    url = create_url(language_ext, url_base)
    model, vocab_size, hidden_size = download_fasttext_data(url)
    real_corrs, random_corrs = \
        run_distance_correlations(model, n_samples, sample_size)

    # Cacheing
    if args.cache_model:
        model_fp = \
            os.path.join(args.out_dir,
                         "{}_model.pickle".format(args.language_ext))
        logging.info("Cacheing model to {}".format(model_fp))
        with open(model_fp, "wb") as fp:
            pickle.dump(model, fp)

    real_corrs_fp = \
        os.path.join(args.out_dir,
                     "{}_real_correlations.npy".format(args.language_ext))
    np.save(real_corrs_fp, real_corrs)

    random_corrs_fp = \
        os.path.join(args.out_dir,
                     "{}_random_correlations.npy".format(args.language_ext))
    np.save(random_corrs_fp, random_corrs)
