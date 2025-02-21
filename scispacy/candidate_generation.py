# import gzip
# import pickle
import os
from typing import Optional, List, Dict, Tuple, NamedTuple, Type
import json
import datetime
from collections import defaultdict
import numpy as np

import bm25s
# import torch
from txtai.pipeline import Tokenizer
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch
import Stemmer

import faiss
from scispacy.file_cache import cached_path
from scispacy.util import get_embedding
import os
import numpy as np
from functools import lru_cache
from gensim.models import KeyedVectors

from scispacy.linking_utils import (
    KnowledgeBase,
    UmlsKnowledgeBase,
    Mesh,
    GeneOntology,
    RxNorm,
    HumanPhenotypeOntology,
)

SUBFOLDER = "202502210921"

class LinkerPaths(NamedTuple):
    """
    Encapsulates all the (possibly remote) paths to data for a scispacy CandidateGenerator.
    index: str
        Path to the approximate nearest neighbours index.
    tfidf_vectorizer: str
        Path to the joblib serialized sklearn TfidfVectorizer.
    tfidf_vectors: str
        Path to the float-16 encoded tf-idf vectors for the entities in the KB.
    concept_aliases_list: str
        Path to the indices mapping concepts to aliases in the index.
    """

    index: str
    tfidf_vectorizer: str
    tfidf_vectors: str
    concept_aliases_list: str

# UmlsLinkerPaths = LinkerPaths(
#     ann_index="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/nmslib_index.bin",  # noqa
#     tfidf_vectorizer="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectorizer.joblib",  # noqa
#     tfidf_vectors="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectors_sparse.npz",  # noqa
#     concept_aliases_list="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/concept_aliases.json",  # noqa
# )

# Test generated artifacts
UmlsLinkerPaths = LinkerPaths(
    index=f"output/{SUBFOLDER}/",  # noqa
    tfidf_vectorizer=f"output/{SUBFOLDER}/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors=f"output/{SUBFOLDER}/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list=f"output/{SUBFOLDER}/concept_aliases.json",  # noqa
)

MeshLinkerPaths = LinkerPaths(
    index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/concept_aliases.json",  # noqa
)

GeneOntologyLinkerPaths = LinkerPaths(
    index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/concept_aliases.json",  # noqa
)

HumanPhenotypeOntologyLinkerPaths = LinkerPaths(
    index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/concept_aliases.json",  # noqa
)

RxNormLinkerPaths = LinkerPaths(
    index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/rxnorm/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/rxnorm/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/rxnorm/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/rxnorm/concept_aliases.json",  # noqa
)

DEFAULT_PATHS: Dict[str, LinkerPaths] = {
    "umls": UmlsLinkerPaths,
    "mesh": MeshLinkerPaths,
    "go": GeneOntologyLinkerPaths,
    "hpo": HumanPhenotypeOntologyLinkerPaths,
    "rxnorm": RxNormLinkerPaths,
}

DEFAULT_KNOWLEDGE_BASES: Dict[str, Type[KnowledgeBase]] = {
    "umls": UmlsKnowledgeBase,
    "mesh": Mesh,
    "go": GeneOntology,
    "hpo": HumanPhenotypeOntology,
    "rxnorm": RxNorm,
}

class MentionCandidate(NamedTuple):
    """
    A data class representing a candidate entity that a mention may be linked to.

    Parameters
    ----------
    concept_id : str, required.
        The canonical concept id in the KB.
    aliases : List[str], required.
        The aliases that caused this entity to be linked.
    similarities : List[float], required.
        The cosine similarities from the mention text to the alias in tf-idf space.

    """

    concept_id: str
    aliases: List[str]
    similarities: List[float]

def load_approximate_nearest_neighbours_index(
    linker_paths: LinkerPaths,
    ef_search: int = 200,
    ):
    """
    Load an approximate nearest neighbours index from disk.

    Parameters
    ----------
    linker_paths: LinkerPaths, required.
        Contains the paths to the data required for the entity linker.
    ef_search: int, optional (default = 200)
        Controls speed performance at query time. Max value is 2000,
        but reducing to around ~100 will increase query speed by an order
        of magnitude for a small performance hit.
    """
    # path = "/home/kgvz782/projects/scispacy/output/202502171706"
    index_path = linker_paths.index
    print(f"Loading index from {index_path}")
    _bm25s_ = bm25s.BM25(backend="numba")
    searcher = _bm25s_.load(index_path, load_corpus=True)
    return searcher

class CandidateGenerator:
    """
    A candidate generator for entity linking to a KnowledgeBase. Currently, two defaults are available:
     - Unified Medical Language System (UMLS).
     - Medical Subject Headings (MESH).

    To use these configured default KBs, pass the `name` parameter, either 'umls' or 'mesh'.

    It uses a sklearn.TfidfVectorizer to embed mention text into a sparse embedding of character 3-grams.
    These are then compared via cosine distance in a pre-indexed approximate nearest neighbours index of
    a subset of all entities and aliases in the KB.

    Once the K nearest neighbours have been retrieved, they are canonicalized to their KB canonical ids.
    This step is required because the index also includes entity aliases, which map to a particular canonical
    entity. This point is important for two reasons:

    1. K nearest neighbours will return a list of Y possible neighbours, where Y < K, because the entity ids
    are canonicalized.

    2. A single string may be an alias for multiple canonical entities. For example, "Jefferson County" may be an
    alias for both the canonical ids "Jefferson County, Iowa" and "Jefferson County, Texas". These are completely
    valid and important aliases to include, but it means that using the candidate generator to implement a naive
    k-nn baseline linker results in very poor performance, because there are multiple entities for some strings
    which have an exact char3-gram match, as these entities contain the same alias string. This situation results
    in multiple entities returned with a distance of 0.0, because they exactly match an alias, making a k-nn
    baseline effectively a random choice between these candidates. However, this doesn't matter if you have a
    classifier on top of the candidate generator, as is intended!

    Parameters
    ----------
    index: FloatIndex
        An nmslib approximate nearest neighbours index.
    tfidf_vectorizer: TfidfVectorizer
        The vectorizer used to encode mentions.
    ann_concept_aliases_list: List[str]
        A list of strings, mapping the indices used in the index to possible KB mentions.
        This is essentially used a lookup between the ann index and actual mention strings.
    kb: KnowledgeBase
        A class representing canonical concepts from the knowledge graph.
    verbose: bool
        Setting to true will print extra information about the generated candidates.
    ef_search: int
        The efs search parameter used in the index. This substantially effects runtime speed
        (higher is slower but slightly more accurate). Note that this parameter is ignored
        if a preconstructed index is passed.
    name: str, optional (default = None)
        The name of the pretrained entity linker to load. Must be one of 'umls' or 'mesh'.
    """

    def __init__(
        self,
        index = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
        ann_concept_aliases_list: Optional[List[str]] = None,
        kb: Optional[KnowledgeBase] = None,
        verbose: bool = False,
        ef_search: int = 200,
        name: Optional[str] = "umls",
    ) -> None:
        if name is not None and any(
            [index, tfidf_vectorizer, ann_concept_aliases_list, kb]
        ):
            raise ValueError(
                "You cannot pass both a name argument and other constuctor arguments."
            )

        # Set the name to the default, after we have checked
        # the compatability with the args above.
        if name is None:
            name = "umls"

        linker_paths = DEFAULT_PATHS.get(name, UmlsLinkerPaths)

        self.bm25_index = index or load_approximate_nearest_neighbours_index(
            linker_paths=linker_paths, ef_search=ef_search
        )

        print(f"Loading concept aliases from {linker_paths.concept_aliases_list}")
        self.concept_aliases = ann_concept_aliases_list or json.load(
            open(cached_path(linker_paths.concept_aliases_list))
        )
        self.faiss_index_path = "data/faiss_index.bin"
        self.faiss_index = self.load_faiss_index()
        
        # ✅ FIX: Ensure self.embedding_cache is properly initialized as an empty dictionary
        self.embedding_cache = {}  # 🚀 Prevents "AttributeError"

        self.kb = kb or DEFAULT_KNOWLEDGE_BASES[name]()
        self.verbose = verbose

        # TODO(Mark): Remove in scispacy v1.0.
        self.umls = self.kb
        self.k_bm25 = 35

    def nmslib_knn_with_zero_vectors(
        self, vectors: numpy.ndarray, k: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        """
        empty_vectors_boolean_flags = numpy.array(vectors.sum(axis=1) != 0).reshape(-1)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
        if self.verbose:
            print(f"Number of empty vectors: {empty_vectors_count}")

        # init extended_neighbors with a list of Nones
        extended_neighbors = numpy.empty(
            (len(empty_vectors_boolean_flags),), dtype=object
        )
        extended_distances = numpy.empty(
            (len(empty_vectors_boolean_flags),), dtype=object
        )

        if vectors.shape[0] - empty_vectors_count == 0:
            return extended_neighbors, extended_distances

        # remove empty vectors before calling `ann_index.knnQueryBatch`
        vectors = vectors[empty_vectors_boolean_flags]

        neighbors, distances = self.index.search_batched(vectors, final_num_neighbors=25)
        
        neighbors = list(neighbors)  # type: ignore
        distances = list(distances)  # type: ignore

        # neighbors need to be converted to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])  # type: ignore
        distances.append([])  # type: ignore
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = numpy.array(
            neighbors, dtype=object
        )[:-1]
        extended_distances[empty_vectors_boolean_flags] = numpy.array(
            distances, dtype=object
        )[:-1]

        return extended_neighbors, extended_distances
        

    # def __call__(
    #     self, mention_texts: List[str], k: int
    # ) -> List[List[MentionCandidate]]:
        """
        Given a list of mention texts, returns a list of candidate neighbors.

        NOTE: Because we include canonical name aliases in the ann index, the list
        of candidates returned will not necessarily be of length k for each candidate,
        because we then map these to canonical ids only.

        NOTE: For a given mention, the returned candidate list might be empty, which implies that
        the tfidf vector for this mention was all zeros (i.e there were no 3 gram overlaps). This
        happens reasonably rarely, but does occasionally.
        Parameters
        ----------
        mention_texts: List[str], required.
            The list of mention strings to generate candidates for.
        k: int, required.
            The number of ann neighbours to look up.
            Note that the number returned may differ due to aliases.

        Returns
        -------
        A list of MentionCandidate objects per mention containing KB concept_ids and aliases
        and distances which were mapped to. Note that these are lists for each concept id,
        because the index contains aliases which are canonicalized, so multiple values may map
        to the same canonical id.
        """
        # if self.verbose:
        #     print(f"Generating candidates for {len(mention_texts)} mentions")

        # start_time = datetime.datetime.now()
        # # tfidf vectorizer crashes on an empty array, so we return early here
        # if mention_texts == []:
        #     return []
        
        # bm25s tokenizer
        # stemmer = Stemmer.Stemmer("english")
        # tokenizer = bm25s.tokenization.Tokenizer(
            # stemmer=stemmer, 
            # stopwords=None,
            # )
        # query_tokenized = tokenizer.tokenize(mention_texts, return_as="string",)
        

        # SPLADE tokenizer and model: FAILS BC WE CANT ACCESS HUGGING FACE
        # tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        # splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        # splade_model.eval()
        # queries_tokenized = [get_splade_sparse_vector(text, tokenizer, splade_model) for text in mention_texts]

        # TXTAI tokenizer
        # tokenizer = Tokenizer()
        # queries_tokenized = [tokenizer(x) for x in mention_texts]

        # end_time = datetime.datetime.now()
        # total_time = end_time - start_time

        # self.verbose= False
        # if self.verbose:
        #     print(f"Mention texts is {mention_texts}")
        #     print(f"Mention texts token is {query_tokenized}")
        #     print(f"Finding neighbors took {total_time.total_seconds()} seconds")
        
        # batch_mention_candidates = []
        # for neighbors, distances in zip(batch_neighbors, batch_distances):
        #     if neighbors is None:
        #         neighbors = []
        #     if distances is None:
        #         distances = []

        #     concept_to_mentions: Dict[str, List[str]] = defaultdict(list)
        #     concept_to_similarities: Dict[str, List[float]] = defaultdict(list)
        #     for neighbor_index, distance in zip(neighbors, distances):
        #         mention = self.ann_concept_aliases_list[neighbor_index]
        #         concepts_for_mention = self.kb.alias_to_cuis[mention]
        #         for concept_id in concepts_for_mention:
        #             concept_to_mentions[concept_id].append(mention)
        #             concept_to_similarities[concept_id].append(1.0 - distance)

        #     mention_candidates = [
        #         MentionCandidate(concept, mentions, concept_to_similarities[concept])
        #         for concept, mentions in concept_to_mentions.items()
        #     ]
        #     mention_candidates = sorted(mention_candidates, key=lambda c: c.concept_id)

        #     batch_mention_candidates.append(mention_candidates)

        # return batch_mention_candidates
    def __call__(self, mention_texts: List[str], k: int) -> List[List[MentionCandidate]]:

        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")

        start_time = datetime.datetime.now()

        if not mention_texts:
            return []

        tokenizer = bm25s.tokenization.Tokenizer()
        query_tokenized = tokenizer.tokenize(mention_texts, return_as="string")

        # **Step 1️⃣: Retrieve top-K BM25 candidates**
        bm25_results, _ = self.bm25_index.retrieve(query_tokenized, k=self.k_bm25)

        batch_mention_candidates = []

        for mention, bm25_candidate_indices in zip(mention_texts, bm25_results):
            # **Step 2️⃣: Get BM25 candidate concepts**
            if bm25_candidate_indices.size == 0:
                batch_mention_candidates.append([])
                continue  # Skip if BM25 found no candidates

            candidate_concepts = [self.concept_aliases[idx] for idx in bm25_candidate_indices]

            # **Step 3️⃣: Get candidate vectors (FAISS input)**
            candidate_vectors = np.array([get_embedding(text) for text in candidate_concepts])

            # **Edge Case: If all vectors are zero (empty embedding), skip FAISS ranking**
            if candidate_vectors.shape[0] == 0 or candidate_vectors.ndim != 2:
                batch_mention_candidates.append([])
                continue

            # **Step 4️⃣: Create a temporary FAISS index with BM25 candidates**
            dimension = candidate_vectors.shape[1]
            temp_faiss_index = faiss.IndexFlatL2(dimension)
            temp_faiss_index.add(candidate_vectors)  # ✅ Only adding BM25 results to FAISS

            # **Step 5️⃣: Convert query to dense embedding**
            query_vector = np.array([get_embedding(mention)])

            # **Step 6️⃣: Search FAISS ONLY within BM25 Candidates**
            distances, indices = temp_faiss_index.search(query_vector, k=min(k, len(candidate_vectors)))

            # **Fix: Ensure indices are valid**
            valid_indices = [idx for idx in indices[0] if idx < len(candidate_concepts)]
            if not valid_indices:
                batch_mention_candidates.append([])
                continue  # Skip if FAISS found no valid candidates

            # **Step 7️⃣: Map FAISS-ranked indices back to concepts**
            ranked_results = [candidate_concepts[idx] for idx in valid_indices]

            # **Step 8️⃣: Format Output with Similarities**
            concept_to_mentions = defaultdict(list)
            concept_to_similarities = defaultdict(list)

            for ranked_concept, distance in zip(ranked_results, distances[0][:len(valid_indices)]):
                concept_ids = self.kb.alias_to_cuis[ranked_concept]
                similarity = 1.0 / (1.0 + distance)  # Convert FAISS L2 distance to similarity

                for concept_id in concept_ids:
                    concept_to_mentions[concept_id].append(ranked_concept)
                    concept_to_similarities[concept_id].append(similarity)

            mention_candidates = [
                MentionCandidate(concept, aliases, concept_to_similarities[concept])
                for concept, aliases in concept_to_mentions.items()
            ]

            batch_mention_candidates.append(mention_candidates)

        end_time = datetime.datetime.now()
        if self.verbose:
            print(f"Processed {len(mention_texts)} mentions in {end_time - start_time}")

        return batch_mention_candidates

    def load_faiss_index(self) -> faiss.Index:
        """
        Load a FAISS index from disk if it exists. Otherwise, create and save a new one.

        Returns
        -------
        faiss.Index
            A FAISS index (either loaded or newly created).
        """
        if os.path.exists(self.faiss_index_path):
            print(f"✅ FAISS index found at {self.faiss_index_path}, loading...")
            faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            print(f"⚠️ FAISS index not found, creating a new one...")

            # Load UMLS concepts and their embeddings
            concept_vectors = np.array([get_embedding(text) for text in self.concept_aliases])
            
            # Create FAISS index with ANN support
            dimension = concept_vectors.shape[1]
            base_index = faiss.IndexHNSWFlat(dimension, 32)  # 🔥 HNSW (Hierarchical Navigable Small World)
            faiss_index = faiss.IndexIDMap2(base_index)  # ✅ Allows storing explicit IDs
            
            # Assign unique IDs to each concept
            ids = np.arange(len(concept_vectors), dtype=np.int64)
            faiss_index.add_with_ids(concept_vectors, ids)
            
            # Save for future use
            print(f"💾 Saving FAISS index to {self.faiss_index_path}")
            faiss.write_index(faiss_index, self.faiss_index_path)

        print(f"🔄 FAISS index contains {faiss_index.ntotal} vectors")
        return faiss_index
        

# def get_splade_sparse_vector(text, tokenizer, splade_model):
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = splade_model(**inputs).logits  # Get token probabilities

#     # Extract token IDs and their importance scores
#     token_ids = inputs.input_ids.squeeze().tolist()
#     token_probs = torch.max(outputs, dim=-1).values.squeeze().tolist()

#     # Keep only important words with a probability threshold
#     sparse_vector = {tokenizer.decode([tid]): prob for tid, prob in zip(token_ids, token_probs) if prob > 0.5}
    
#     return sparse_vector