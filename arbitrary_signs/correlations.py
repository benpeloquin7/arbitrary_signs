"""correlations.py"""

from collections import defaultdict
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


def language_typ_from_url(url):
    return url[-6:-4]

def fasttext_fp_creator(language, out_dir):
    return os.path.join(out_dir, "{}_data.pickle".format(language))


def download_fasttext_data(url, verbose=True, check_dir=True, out_dir=None):
    """Returns tuple of (dict, int, int)"""

    # First check to see if we've cached anything (for efficiency)
    if check_dir and out_dir is not None:
        lang = language_typ_from_url(url)
        check_fp = fasttext_fp_creator(lang, out_dir)
        if os.path.exists(check_fp):
            with open(check_fp, "rb") as fp:
                data = pickle.load(fp)
                vocab_size = len(data.keys())
                hidden_size = len(data[list(data.keys())[0]])
            if verbose:
                logging.info("Using cached data from {}".format(check_fp))
            return data, vocab_size, hidden_size

    # Otherwise load data anew
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
        try:
            word = elements[0].decode('utf-8')
        except:
            import pdb; pdb.set_trace();
        vec = np.array(list(map(float, elements[1:])))
        data[word] = vec
        pbar.update()
    pbar.close()

    return data, vocab_size, hidden_size


def cache_fasttext_data(data, language, out_dir, verbose=True):
    fasttext_data_fp = fasttext_fp_creator(language, out_dir)
    if os.path.exists(fasttext_data_fp):
        pass
    else:
        if verbose:
            logging.info(
                "Cacheing {} language data to {}".format(language,
                                                         fasttext_data_fp))
        with open(fasttext_data_fp, "wb") as fp:
            pickle.dump(data, fp)


def get_vocab_sample(model, sample_size=500):
    words = np.array(list(model.keys()))
    vecs = np.array(list(model.values()))
    idxs = np.random.choice(range(len(words)), size=sample_size)
    words_sample = words[idxs]
    vecs_sample = vecs[idxs]
    return words_sample, vecs_sample


def create_pairs(model, sample_size=500):
    words, vecs = get_vocab_sample(model, sample_size)
    words_shuffle, vecs_shuffle = words.copy(), vecs.copy()
    new_idxs = np.random.permutation(range(sample_size))
    words_shuffle, vecs_shuffle = \
        words_shuffle[new_idxs], vecs_shuffle[new_idxs]
    return (words, words_shuffle), (vecs, vecs_shuffle)


def get_dists(v1, v2, dist_fn):
    # Note that zipping creates a tuple so we index.
    return list(map(lambda x: dist_fn(x[0], x[1]), zip(v1, v2)))


def get_correlations(dists1, dists2):
    return np.corrcoef(dists1, dists2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language-ext", type=str, default="en",
                        help="Language extension [Default: 'en'].")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples [Default: 100].")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Sample size [Default: 500].")
    parser.add_argument("--cache-data", action='store_true', default=True,
                        help="Pickle data file [Default: True].")
    parser.add_argument("--out-dir", type=str, default="./outputs",
                        help="Output directory [Default: ./outputs].")

    args = parser.parse_args()

    language_ext = args.language_ext
    n_samples = args.n_samples
    sample_size = args.sample_size

    make_safe_dir(args.out_dir)

    url = create_url(language_ext, url_base)
    data, vocab_size, hidden_size = \
        download_fasttext_data(url, check_dir=True, out_dir=args.out_dir)

    cache_fasttext_data(data, language_ext, args.out_dir)

    dists = defaultdict(dict)
    sim_type = "real"
    for batch in tqdm.tqdm(range(args.n_samples)):
        (w1, w2), (v1, v2) = create_pairs(data, args.sample_size)
        # Real vals
        word_dists = get_dists(w1, w2, levenshtein)
        vec_dists = get_dists(v1, v2, cosine)
        dists[batch]["real_word_dists"] = word_dists
        dists[batch]["real_vector_dists"] = vec_dists
        # Permuted vals
        shuffle_idxs = np.random.choice(range(len(w1)), args.sample_size)
        dists[batch]["random_word_dists"] = \
            get_dists(w1[shuffle_idxs], w2, levenshtein)
        # Store words
        dists[batch]["real_word_pairs"] = list(zip(w1, w2))
        dists[batch]["shuffle_idxs"] = shuffle_idxs

    # FP is language-#samples-samplesize
    # Example:
    #     >>> 'ru_1000_10000.pickle'
    dists_fp = os.path.join(args.out_dir, "{}_{}_{}_dists.pickle".format(
        args.language_ext, args.n_samples, args.sample_size))
    logging.info("Dumping simulations to {}".format(dists_fp))
    with open(dists_fp, "wb") as fp:
        pickle.dump(dists, fp)