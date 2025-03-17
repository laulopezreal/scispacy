import os
import faiss
from packaging.version import Version
import spacy
import scipy
from spacy.language import Language
from spacy.tokens import Doc

from scispacy.custom_sentence_segmenter import pysbd_sentencizer
from scispacy.custom_tokenizer import combined_rule_tokenizer


def save_model(nlp: Language, output_path: str):
    nlp.to_disk(output_path)


def create_combined_rule_model() -> Language:
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(pysbd_sentencizer, first=True)
    return nlp


def scipy_supports_sparse_float16() -> bool:
    # https://github.com/scipy/scipy/issues/7408
    return Version(scipy.__version__) < Version("1.11")


class WhitespaceTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        # All tokens 'own' a subsequent space character in
        # this tokenizer. This is a technicality and probably
        # not that interesting.
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

import os
import numpy as np
from functools import lru_cache
from gensim.models import KeyedVectors

# 🔥 Load FastText model only ONCE and store globally
fasttext_model = None

def load_fasttext():
    """ Load FastText model once into memory. """
    global fasttext_model
    model_path = "/home/kgvz782/projects/scispacy/data/models/fasttext/fasttext.model"

    if fasttext_model is None:
        if os.path.exists(model_path):
            print(f"✅ Loading cached FastText model from {model_path}")
            fasttext_model = KeyedVectors.load(model_path, mmap='r')
        else:
            print(f"⚡ Loading FastText vectors from raw file...")            
            fasttext_model = KeyedVectors.load_word2vec_format(
                "data/models/fasttext/cc.en.300.vec", binary=False
            )
            fasttext_model.save(model_path)
            print(f"✅ Model saved for faster future use.")
    
    return fasttext_model


# ✅ Use LRU cache to store embeddings & avoid recomputation
@lru_cache(maxsize=100000)  # Stores embeddings for up to 100,000 texts
def get_embedding(text: str):
    """
    Returns a dense embedding for a given text.
    - Uses preloaded FastText model.
    - Caches embeddings for fast repeated queries.
    """
    fasttext = load_fasttext()  # Ensures model is loaded only once
    words = text.split()

    # 🔥 Vectorize faster: Avoid looping, use NumPy
    word_vectors = [fasttext[word] for word in words if word in fasttext]
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word embeddings
    else:
        return np.zeros(fasttext.vector_size)  # Return zero vector if empty


def hybrid_retrieval(query, bm25_index, tokenizer, get_embedding, umls_concepts, k_bm25=40, k_faiss=10):
    """
    Hybrid retrieval using BM25 (sparse) for recall and FAISS (dense) for ranking.
    """
    # 1️⃣ Retrieve top-K candidates from BM25
    query_tokenized = tokenizer.tokenize([query])
    bm25_results, _ = bm25_index.retrieve(query_tokenized, k=k_bm25)
    candidate_concepts = [umls_concepts[idx] for idx in bm25_results]  # Candidate texts

    # 2️⃣ Convert query to dense embedding
    query_vector = np.array([get_embedding(query)])

    # 3️⃣ Convert BM25 candidates into dense embeddings
    candidate_vectors = np.array([get_embedding(text) for text in candidate_concepts])

    # 4️⃣ Create a temporary FAISS index with only BM25 candidates
    dimension = candidate_vectors.shape[1]
    temp_faiss_index = faiss.IndexFlatL2(dimension)
    temp_faiss_index.add(candidate_vectors)  # Use only BM25 candidates

    # 5️⃣ Retrieve top-K from FAISS (semantic re-ranking)
    distances, indices = temp_faiss_index.search(query_vector, k_faiss)

    # 6️⃣ Map indices to ranked BM25 candidates
    ranked_results = [candidate_concepts[idx] for idx in indices[0]]

    return ranked_results