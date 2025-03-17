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
        self.allowed_faiss_models = ["BFL2", "HNSW"]
        
        self.faiss_model = "HNSW"
        if self.faiss_model not in self.allowed_faiss_models:
            raise ValueError(
                f"Invalid FAISS model: {self.faiss_model}. Choose from {self.allowed_faiss_models}"
            )
        self.faiss_index_path = f"data/{self.faiss_model}_faiss_index.bin"  
        print(f"🔍 Using FAISS model: {self.faiss_model}")
        # self.faiss_index = self.load_faiss_index()
        
        # ✅ FIX: Ensure self.embedding_cache is properly initialized as an empty dictionary
        self.embedding_cache = {}  # 🚀 Prevents "AttributeError"

        self.kb = kb or DEFAULT_KNOWLEDGE_BASES[name]()
        self.verbose = verbose

        # TODO(Mark): Remove in scispacy v1.0.
        self.umls = self.kb
        self.k_bm25 = 35
        self.k_faiss = 15

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
        
        
    def __call__(self, mention_texts: List[str], k: int) -> List[List[MentionCandidate]]:

        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")
            
        if not mention_texts:
            return []

        tokenizer = bm25s.tokenization.Tokenizer()
        query_tokenized = tokenizer.tokenize(mention_texts, return_as="string")

        # **Step 1️⃣: Retrieve top-K BM25 candidates**
        bm25_results, _ = self.bm25_index.retrieve(query_tokenized, k=self.k_bm25, backend_selection="numba")
        if self.verbose:
            print(f"Generated BM25 {len(self.k_bm25,)} candidates: {bm25_results}")
        
        return self.temporary_faiss_index(mention_texts, bm25_results, self.k_faiss)
            
    def index_faiss_persistent(self, mention_texts, bm25_results):
        start_time = datetime.datetime.now()
        batch_mention_candidates = []
        for mention, bm25_candidate_indices in zip(mention_texts, bm25_results):
            # Step 1️⃣: Get BM25 candidate vectors
            candidate_concepts = [self.concept_aliases[idx] for idx in bm25_candidate_indices]
            candidate_vectors = np.array([get_embedding(text) for text in candidate_concepts])
            
            if candidate_vectors.shape[0] == 0 or candidate_vectors.ndim != 2:
                batch_mention_candidates.append([])
                continue  # Skip if no valid vectors

            # Step 2️⃣: Get FAISS Indices for BM25 Candidates
            candidate_indices = np.array(bm25_candidate_indices, dtype=np.int64)
            candidate_ids = np.array(bm25_candidate_indices, dtype=np.int64)  # ✅ Convert BM25 indices to NumPy array


            # Step 3️⃣: Restrict FAISS Search to BM25 Candidates
            distances, indices = self.faiss_index.search(query_vector, k=min(self.k_faiss, len(candidate_ids)))

            # sub_faiss_index = faiss.IndexIDMap2(self.faiss_index)  # ✅ Wrap FAISS index
            # sub_faiss_index.add_with_ids(candidate_vectors, candidate_indices)  # ✅ Restrict to BM25
            
            # **Step 4️⃣: Filter results only to BM25 candidates**
            filtered_indices = [idx for idx in indices[0] if idx in candidate_ids]
            if not filtered_indices:
                batch_mention_candidates.append([])
                continue  # Skip if FAISS found no valid candidates

            # **Step 5️⃣: Map FAISS-ranked indices back to concepts**
            ranked_results = [candidate_concepts[np.where(candidate_ids == idx)[0][0]] for idx in filtered_indices]

            # **Step 6️⃣: Format Output with Similarities**
            concept_to_mentions = defaultdict(list)
            concept_to_similarities = defaultdict(list)

            # Step 4️⃣: Search FAISS only within BM25 Candidates
            query_vector = np.array([get_embedding(mention)])
            distances, indices = sub_faiss_index.search(query_vector, k=min(self.k_faiss, len(candidate_indices)))

            # Step 5️⃣: Map Results Back to Concepts
            ranked_results = [candidate_concepts[idx] for idx in indices[0] if idx < len(candidate_concepts)]

            # Step 6️⃣: Format Output
            concept_to_mentions = defaultdict(list)
            concept_to_similarities = defaultdict(list)
            for ranked_concept, distance in zip(ranked_results, distances[0]):
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
        
    def temporary_faiss_index(self, mention_texts, bm25_results, k):
        start_time = datetime.datetime.now()
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
            
            # **Step 5️⃣: Convert query to dense embedding**
            query_vector = np.array([get_embedding(mention)])

            # **Step 4️⃣: Create a temporary FAISS index with BM25 candidates**
            dimension = candidate_vectors.shape[1]
            # M = 32  # Number of neighbors in HNSW graph (typical choice)
            # temp_faiss_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
            
            # Using FlatL2 for brute-force search
            temp_faiss_index = faiss.IndexFlatL2(dimension)

            # temp_faiss_index.add(candidate_vectors)  # ✅ Only adding BM25 results to FAISS
 
            # **Step 6️⃣: Search FAISS ONLY within BM25 Candidates**
            distances, indices = temp_faiss_index.search(query_vector, k=min(self.k_faiss, len(candidate_vectors)))
            # distances, indices = temp_faiss_index.search(query_vector, k=min(k, len(candidate_vectors)))

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
        Load or create FAISS index using HNSW for faster ANN search.
        """
        if os.path.exists(self.faiss_index_path):
            print(f"✅ Loading FAISS index from {self.faiss_index_path}")
            faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            print(f"⚠️ FAISS index  using {self.faiss_model} not found, creating a new one...")
            # Load UMLS concepts and their embeddings
            concept_vectors = np.array([get_embedding(text) for text in self.concept_aliases])
            dimension = concept_vectors.shape[1]
            if self.faiss_model == "BFL2":
                base_index = faiss.IndexFlatL2(dimension, 32)
            elif self.faiss_model == "HNSW":
                # ✅ Use FAISS HNSW for Approximate Nearest Neighbors (faster than brute-force L2)
                base_index = faiss.IndexHNSWFlat(dimension, 32)
               
            faiss_index = faiss.IndexIDMap2(base_index)  # ✅ Allows explicit ID mapping
            
            # Assign unique IDs to each concept
            ids = np.arange(len(concept_vectors), dtype=np.int64)
            faiss_index.add_with_ids(concept_vectors, ids)
            
            # Save for future use
            print(f"💾 Saving FAISS index to {self.faiss_index_path}")
            faiss.write_index(faiss_index, self.faiss_index_path)
            
            
        # **Fix: Only wrap if not already an IndexIDMap2**
        if not isinstance(faiss_index, faiss.IndexIDMap2):
            print("⚠️ Wrapping FAISS index in IndexIDMap2...")
            faiss_index = faiss.IndexIDMap2(faiss_index)

        print(f"🔄 FAISS index contains {faiss_index.ntotal} vectors")
        return faiss_index
