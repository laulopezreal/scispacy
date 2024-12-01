# import gzip
# import pickle
from typing import Optional, List, Dict, Tuple, NamedTuple, Type
import json
import datetime
from collections import defaultdict

import os
import scann
import scipy
import numpy
import joblib
import scipy.sparse
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
# import nmslib
from nmslib.dist import FloatIndex
# from pynndescent import NNDescent
# from scispacy.util import scipy_supports_sparse_float16
from scispacy.file_cache import cached_path
from scispacy.linking_utils import (
    KnowledgeBase,
    UmlsKnowledgeBase,
    Mesh,
    GeneOntology,
    RxNorm,
    HumanPhenotypeOntology,
)

SUBFOLDER = "202411301949"

class LinkerPaths(NamedTuple):
    """
    Encapsulates all the (possibly remote) paths to data for a scispacy CandidateGenerator.
    ann_index: str
        Path to the approximate nearest neighbours index.
    tfidf_vectorizer: str
        Path to the joblib serialized sklearn TfidfVectorizer.
    tfidf_vectors: str
        Path to the float-16 encoded tf-idf vectors for the entities in the KB.
    concept_aliases_list: str
        Path to the indices mapping concepts to aliases in the index.
    """

    ann_index: str
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
    ann_index=f"/home/kgvz782/scispacy_output/{SUBFOLDER}/index/",  # noqa
    tfidf_vectorizer=f"/home/kgvz782/scispacy_output/{SUBFOLDER}/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors=f"/home/kgvz782/scispacy_output/{SUBFOLDER}/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list=f"/home/kgvz782/scispacy_output/{SUBFOLDER}/concept_aliases.json",  # noqa
)

MeshLinkerPaths = LinkerPaths(
    ann_index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/mesh/concept_aliases.json",  # noqa
)

GeneOntologyLinkerPaths = LinkerPaths(
    ann_index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/go/concept_aliases.json",  # noqa
)

HumanPhenotypeOntologyLinkerPaths = LinkerPaths(
    ann_index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/nmslib_index.bin",  # noqa
    tfidf_vectorizer="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/hpo/concept_aliases.json",  # noqa
)

RxNormLinkerPaths = LinkerPaths(
    ann_index="https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/linkers/2023-04-23/rxnorm/nmslib_index.bin",  # noqa
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
) -> FloatIndex:
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
    concept_alias_tfidfs = scipy.sparse.load_npz(
        cached_path(linker_paths.tfidf_vectors)
    ).astype(numpy.float32)
    # ann_index = nmslib.init(
    #     method="hnsw",
    #     space="cosinesimil_sparse",
    #     data_type=nmslib.DataType.SPARSE_VECTOR,
    # )
    # ann_index.addDataPointBatch(concept_alias_tfidfs)
    # ann_index.loadIndex(cached_path(linker_paths.ann_index))
    # query_time_params = {"efSearch": ef_search}
    # ann_index.setQueryTimeParams(query_time_params)
    # with gzip.open(cached_path(linker_paths.ann_index), "wb") as f:
    # ann_index = joblib.load(cached_path(linker_paths.ann_index))

    # SCANN ANN index load
    print(f"Using ANN index at {cached_path(linker_paths.ann_index)}")
    ann_index_searcher = scann.scann_ops_pybind.load_searcher(cached_path(linker_paths.ann_index))
    # neighbors, distances = another_searcher.search_batched(dataset, final_num_neighbors=25)
    return ann_index_searcher

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
    ann_index: FloatIndex
        An nmslib approximate nearest neighbours index.
    tfidf_vectorizer: TfidfVectorizer
        The vectorizer used to encode mentions.
    ann_concept_aliases_list: List[str]
        A list of strings, mapping the indices used in the ann_index to possible KB mentions.
        This is essentially used a lookup between the ann index and actual mention strings.
    kb: KnowledgeBase
        A class representing canonical concepts from the knowledge graph.
    verbose: bool
        Setting to true will print extra information about the generated candidates.
    ef_search: int
        The efs search parameter used in the index. This substantially effects runtime speed
        (higher is slower but slightly more accurate). Note that this parameter is ignored
        if a preconstructed ann_index is passed.
    name: str, optional (default = None)
        The name of the pretrained entity linker to load. Must be one of 'umls' or 'mesh'.
    """

    def __init__(
        self,
        ann_index = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
        ann_concept_aliases_list: Optional[List[str]] = None,
        kb: Optional[KnowledgeBase] = None,
        verbose: bool = False,
        ef_search: int = 200,
        name: Optional[str] = "umls",
    ) -> None:
        if name is not None and any(
            [ann_index, tfidf_vectorizer, ann_concept_aliases_list, kb]
        ):
            raise ValueError(
                "You cannot pass both a name argument and other constuctor arguments."
            )

        # Set the name to the default, after we have checked
        # the compatability with the args above.
        if name is None:
            name = "umls"

        linker_paths = DEFAULT_PATHS.get(name, UmlsLinkerPaths)

        print(f"Loading ANN index from {linker_paths.ann_index}")
        self.ann_index = ann_index or load_approximate_nearest_neighbours_index(
            linker_paths=linker_paths, ef_search=ef_search
        )

        print(f"Loading TFIDF vectorizer from {linker_paths.tfidf_vectorizer}")
        self.vectorizer = tfidf_vectorizer or joblib.load(
            cached_path(linker_paths.tfidf_vectorizer)
        )

        print(f"Loading ANN concept aliases  from {linker_paths.concept_aliases_list}")
        self.ann_concept_aliases_list = ann_concept_aliases_list or json.load(
            open(cached_path(linker_paths.concept_aliases_list))
        )

        self.kb = kb or DEFAULT_KNOWLEDGE_BASES[name]()
        self.verbose = verbose

        # TODO(Mark): Remove in scispacy v1.0.
        self.umls = self.kb

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

        # NMSLIB VERSION call `knnQueryBatch` to get neighbors
        # original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)

        # PYNNDESCENT VERSION query to get neighbors
        # original_neighbours = self.ann_index.query(vectors, k=k)

        neighbors, distances = self.ann_index.search_batched(vectors, final_num_neighbors=25)

        # neighbors, distances = zip(
        #     *[(x[0].tolist(), x[1].tolist()) for x in original_neighbours]
        # )
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

    def __call__(
        self, mention_texts: List[str], k: int
    ) -> List[List[MentionCandidate]]:
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
        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")

        # tfidf vectorizer crashes on an empty array, so we return early here
        if mention_texts == []:
            return []

        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()
        if self.verbose:
            print(f"Initial TFIDFs data type is {type(tfidfs)}")
            print(f"Initial TFIDFs data shape is {tfidfs.shape}")


        # Apply Truncated SVD
        if self.verbose:
            print("Dimensionality reduction using Truncated Singular Value Decomposition (LatentSemantic Analysis)")
        from sklearn.decomposition import TruncatedSVD
        # Step 1: Apply dimensionality reduction to query
        import joblib

        svd = joblib.load(f"/home/kgvz782/scispacy_output/{SUBFOLDER}/svd_model.joblib")
        tfidfs_reduced_query = svd.transform(tfidfs)

        # Step 2: Normalize the query vectors if using cosine similarity
        from sklearn.preprocessing import normalize
        tfidfs_reduced_query = normalize(tfidfs_reduced_query, axis=1)
        if self.verbose:
            print(f"Converted TGIDFs data shape {tfidfs_reduced_query.shape}")

        # Convert to NumPy float32 array -> 
        # TODO: At this point tfidfs_reduced_query is already a numpy array, and the 2 lines below are not needed
        # import numpy as np
        # tfidfs_dense_array = tfidfs_reduced_query.toarray().astype(np.float32)
        if self.verbose:
            print(f"Numpy Converted TGIDFs data type is {type(tfidfs_reduced_query)}")
            print(f"Numpy Converted TGIDFs data shape {tfidfs_reduced_query.shape}")

        # Step 3: Perform the search
        # assert tfidfs_reduced_query.shape[1] == tfidf_reduced.shape[1], "Dimensionality mismatch!"

        batch_neighbors, batch_distances = self.ann_index.search_batched(
            tfidfs_reduced_query, final_num_neighbors=25
        )

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        # batch_neighbors, batch_distances = self.ann_index.search_batched(tfidfs_dense_array, final_num_neighbors=25)
        # batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        if self.verbose:
            print(f"Finding neighbors took {total_time.total_seconds()} seconds")
        batch_mention_candidates = []
        for neighbors, distances in zip(batch_neighbors, batch_distances):
            if neighbors is None:
                neighbors = []
            if distances is None:
                distances = []

            concept_to_mentions: Dict[str, List[str]] = defaultdict(list)
            concept_to_similarities: Dict[str, List[float]] = defaultdict(list)
            for neighbor_index, distance in zip(neighbors, distances):
                mention = self.ann_concept_aliases_list[neighbor_index]
                concepts_for_mention = self.kb.alias_to_cuis[mention]
                for concept_id in concepts_for_mention:
                    concept_to_mentions[concept_id].append(mention)
                    concept_to_similarities[concept_id].append(1.0 - distance)

            mention_candidates = [
                MentionCandidate(concept, mentions, concept_to_similarities[concept])
                for concept, mentions in concept_to_mentions.items()
            ]
            mention_candidates = sorted(mention_candidates, key=lambda c: c.concept_id)

            batch_mention_candidates.append(mention_candidates)

        return batch_mention_candidates


def create_tfidf_ann_index(
    out_path: str, kb: Optional[KnowledgeBase] = None, test_mode: bool = False
) -> Tuple[List[str], TfidfVectorizer, FloatIndex]:
    """
    Build tfidf vectorizer and ann index.

    Parameters
    ----------
    out_path: str, required.
        The path where the various model pieces will be saved.
    kb : KnowledgeBase, optional.
        The kb items to generate the index and vectors for.

    """
    # if not scipy_supports_sparse_float16():
    #     raise RuntimeError(
    #         "This function requires scipy<1.11, which only runs on Python<3.11."
    #     )

    # Create a subfolder to save the linker artifacts
    #  Format datetime as YYYYmmddhhmm  
    date_subfolder = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_path = f"{out_path}{date_subfolder}"

    print(f"Creating subfolder to save the outputs at {output_path}")
    os.makedirs(output_path, exist_ok=True,)

    tfidf_vectorizer_path = f"{output_path}/tfidf_vectorizer.joblib"
    # ann_index_path = f"{output_path}/ann_index.npz"
    tfidf_vectors_path = f"{output_path}/tfidf_vectors_sparse.npz"
    umls_concept_aliases_path = f"{output_path}/concept_aliases.json"

    cached_data = "/home/kgvz782/.scispacy/datasets/"
    umls_file = cached_data + "d5e593bc2d8adeee7754be423cd64f5d331ebf26272074a2575616be55697632.0660f30a60ad00fffd8bbf084a18eb3f462fd192ac5563bf50940fc32a850a3c.umls_2022_ab_cat0129.jsonl"
    umls_types_file = cached_data + "21a1012c532c3a431d60895c509f5b4d45b0f8966c4178b892190a302b21836f.330707f4efe774134872b9f77f0e3208c1d30f50800b3b39a6b8ec21d9adf1b7.umls_semantic_type_tree.tsv"
    if kb:
        kb = kb
    elif os.path.exists(umls_file) and os.path.exists(umls_types_file):
        kb = UmlsKnowledgeBase(file_path=umls_file, types_file_path=umls_types_file)
    else:
        kb = UmlsKnowledgeBase()

    # NMSLIB hyperparameters (very important)
    # guide: https://github.com/nmslib/nmslib/blob/master/manual/methods.md
    # Default values resulted in very low recall.

    # set to the maximum recommended value. Improves recall at the expense of longer indexing time.
    # We use the HNSW (Hierarchical Navigable Small World Graph) representation which is constructed
    # by consecutive insertion of elements in a random order by connecting them to M closest neighbours
    # from the previously inserted elements. These later become bridges between the network hubs that
    # improve overall graph connectivity. (bigger M -> higher recall, slower creation)
    # For more details see:  https://arxiv.org/pdf/1603.09320.pdf?
    m_parameter = 100

    # `C` for Construction. Set to the maximum recommended value
    # Improves recall at the expense of longer indexing time
    construction = 2000
    num_threads = 60  # set based on the machine
    index_params = {
        "M": m_parameter,
        "indexThreadQty": num_threads,
        "efConstruction": construction,
        "post": 0,
    }
    
    # Get concept aliases from Knowledge b=Base
    concept_aliases = list(kb.alias_to_cuis.keys())
    initial_n = len(concept_aliases)

    if test_mode:
        concept_aliases = concept_aliases[0:4000000] 
        print(f"Test mode enabled: reducing concept aliases from {initial_n} to {len(concept_aliases)} for testing")
        

    vector_exists = os.path.exists(tfidf_vectorizer_path)
    if not vector_exists:
        print(f"TFIDF vectorizernot NOT found in {tfidf_vectorizer_path}")
    elif vector_exists:
        print(f"TFIDF vectorizer found in {tfidf_vectorizer_path}")
    
    # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
    # resulting vectors using float16, meaning they take up half the memory on disk --- 
    # TODO: Regarding the above BUT THEN YOU INTRODUCE A INCOPATIBILIYY WITH SPICY AND PYTHON>3.11.

    #  Unfortunately we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
    # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408

    # Fitting TFIDF vectorizer
    fitting_start = datetime.datetime.now()
    print(f"Fitting TFIDF vectorizer on {len(concept_aliases)} aliases")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=numpy.float32
    )
    concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
    fitting_end = datetime.datetime.now()
    fitting_time = fitting_end - fitting_start
    print(f"Fitting the tfidf vectorizer took {fitting_time.total_seconds()} seconds")

    # Saving TFIDF vectorizer
    import joblib
    saving_start = datetime.datetime.now()
    print(f"Saving tfidf vectorizer to {tfidf_vectorizer_path}")
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    saving_end = datetime.datetime.now()
    saving_time = saving_end - saving_start
    print(f"Saving the tfid vectorizer took {saving_time.total_seconds()} seconds")

     # Find zero vectors
    print("Finding empty (all zeros) tfidf vectors")
    empty_tfidfs_boolean_flags = numpy.array(
        concept_alias_tfidfs.sum(axis=1) != 0
    ).reshape(-1)
    number_of_non_empty_tfidfs = sum(empty_tfidfs_boolean_flags == False)  # noqa: E712
    total_number_of_tfidfs = numpy.size(concept_alias_tfidfs, 0)
    print(f"TFIDF vectors dimmension: {concept_alias_tfidfs.shape}")

    # Remove zero vectors
    print(
        f"Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty"
    )
    # remove empty tfidf vectors, otherwise nmslib will crash
    concept_aliases = [
        alias
        for alias, flag in zip(concept_aliases, empty_tfidfs_boolean_flags)
        if flag
    ]
    concept_alias_tfidfs = concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    assert len(concept_aliases) == numpy.size(concept_alias_tfidfs, 0)
    print(f"TFIDF vectors dimmension after removing empty tfidf vectors: {concept_alias_tfidfs.shape}")


    # Save removed vectors
    print(
        f"Saving list of concept ids and tfidfs vectors to {umls_concept_aliases_path} and {tfidf_vectors_path}"
    )
    json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))
    scipy.sparse.save_npz(  
        tfidf_vectors_path, concept_alias_tfidfs.astype(numpy.float32)
    )

    # TODO: Add documentation -> Why use PyNNDescent?
    # PyNNDescent provides fast approximate nearest neighbor queries. 
    # The ann-benchmarks system puts it solidly in the mix of top performing ANN libraries:

    # Dimensionality reduction using Truncated Singular Value Decomposition (LatentSemantic Analysis)
    # Apply Truncated SVD
    print("Dimensionality reduction using Truncated Singular Value Decomposition (LatentSemantic Analysis)")
    from sklearn.decomposition import TruncatedSVD
    n_components = 2000  # Number of dimensions to reduce to
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reduced = svd.fit_transform(concept_alias_tfidfs)

    # Save the SVD instance for reuse
    import joblib
    joblib.dump(svd, f"{output_path}/svd_model.joblib")

    print("Original shape:", concept_alias_tfidfs.shape)
    print("Reduced shape:", tfidf_reduced.shape)
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance * 100:.2f}%")
    import sys
    print(f"Reduced tfidf size: {sys.getsizeof(tfidf_reduced) / (1024 ** 2):.2f} MB")

    # Not needede bc. dimensionality reduction
    # # Convert tfidf vectors to Compressed Sparse Row matrix data type (CSR)
    # conversion_start = datetime.datetime.now()
    # print("Converting from numpy array to Compressed Sparse Row format")
    # sparse_matrix_csr = scipy.sparse.csr_matrix(tfidf_reduced)

    # # del concept_alias_tfidfs

    # print(f"\nCSR sparse matrix dimmension: {sparse_matrix_csr.shape}")
    # # print(type(sparse_matrix_csr))
    # # print(sparse_matrix_csr.data)
    # # print(sparse_matrix_csr.indices.size)

    # sparse_matrix_memory =     (
    #     sparse_matrix_csr.data.nbytes
    #     + sparse_matrix_csr.indices.size
    #     + sparse_matrix_csr.indptr.nbytes
    # ) / 1024**2
    # print(f"CSR memory is {sparse_matrix_memory} Mb")
    
    # # Remove rows with no data or low density:
    # print("Remove rows with no data or low density:")
    # sparse_matrix_csr = sparse_matrix_csr[sparse_matrix_csr.getnnz(1) > 0]
    # # Monitor the density of the matrix:
    # density = sparse_matrix_csr.nnz / (sparse_matrix_csr.shape[0] * sparse_matrix_csr.shape[1])
    # print(f"Matrix density: {density:.4f}")
    # conversion_ends = datetime.datetime.now()
    # conversion_time = conversion_ends - conversion_start
    # print(f"Converting the numpy array to CSR format took {conversion_time}")

    # SCANN by google
    # 1. Convert to NumPy Array
    import numpy as np

    # Assuming 'tfidf_reduced' is your reduced-dimensional TF-IDF matrix
    tfidf_reduced = np.asarray(tfidf_reduced, dtype=np.float32)

    # 2. Normalize Vectors (If Using Cosine Similarity)
    # For cosine similarity, normalize your vectors to unit length.

    from sklearn.preprocessing import normalize
    tfidf_reduced = normalize(tfidf_reduced, axis=1)

    # 3. Build the ScaNN Index
    # ScaNN provides a flexible API to configure and build your index.

    import scann

    # Number of nearest neighbors to retrieve
    k = 10

    # Build the ScaNN index
    ann_searcher = scann.scann_ops_pybind.builder(
        tfidf_reduced, num_neighbors=k, distance_measure="dot_product"
    ).tree(
        num_leaves=1000,  # Adjust based on dataset size
        num_leaves_to_search=100,  # Trade-off between speed and accuracy
        training_sample_size=250000  # Number of training samples for the partitioning tree
    ).score_ah(
        2,  # Number of bits per dimension (Anisotropic Hashing)
        anisotropic_quantization_threshold=0.2
    ).reorder(
        100  # Number of top candidates to reorder for exact distance computation
    ).build()


    # another_searcher = scann.scann_ops_pybind.load_searcher(INDEX_DIR)
    # neighbors, distances = another_searcher.search_batched(dataset, final_num_neighbors=25)

    # Explanation of Parameters:
    # num_neighbors=k: Number of nearest neighbors to find.
    # distance_measure="dot_product": Suitable for normalized vectors (cosine similarity).
    # num_leaves: Number of leaves in the partitioning tree; higher values can improve recall.
    # num_leaves_to_search: Number of leaves to search over; increasing improves accuracy but slows down search.
    # training_sample_size: Number of points used to train the partitioning tree.
    # score_ah(2, ...): Configures Asymmetric Hashing for faster search.
    # reorder(100): Re-ranks the top 100 candidates using exact distances to improve accuracy.

    # PYNNDESCENT
    # Fitting ANN on concept aliases
    # print(f"Fitting ANN PyNNDescent index on {len(concept_aliases)} aliases")
    # fitting_ann_index_start = datetime.datetime.now()
    # ann_index = NNDescent(
    #     sparse_matrix_csr, 
    #     metric='cosine', 
    #     low_memory=True, 
    #     n_trees=5,
    #     verbose=True,
    #     compressed=True,
    #     )
    # ann_index = NNDescent(
    #     tfidf_reduced, 
    #     metric='cosine', 
    #     low_memory=True, 
    #     n_trees=5,
    #     verbose=True,
    #     compressed=True,
    #     )
    # fitting_ann_index_end = datetime.datetime.now()
    # fitting_ann_index_time = fitting_ann_index_end - fitting_ann_index_start
    # print(f"Fitting PyNNDescent index took {fitting_ann_index_time.total_seconds()} seconds")

    # import sys
    # print(f"Index size: {sys.getsizeof(ann_index) / (1024 ** 2):.2f} MB")
    
    # Saving ANN index
    saving_start = datetime.datetime.now()
    INDEX_DIR = f'{output_path}/index'
    os.makedirs(INDEX_DIR, exist_ok=True)
    ann_searcher.serialize(INDEX_DIR) # store the scann_module 
    # print(f"Attempting to save the ANN index to {ann_index_path} using JOBLIB w/ compression 3")

    # This below crashes!
    # joblib.dump(ann_index, ann_index_path)

    # Alternative?
    # index = joblib.load("/home/kgvz782/projects/scispacy/test_output/nmslib_index.bin")
    # index = scipy.sparse.load_npz("/home/kgvz782/projects/scispacy/test_output/ann_index.npz")

    # Let's try joblib
    # Save the PyNNDescent index to a file
    # Oom error
    # joblib.dump(ann_index, ann_index_path, compress=3)

    # oom error
    # joblib.dump(ann_index, ann_index_path)

    # # Pickle
    # print(f"Attempting to save the ANN index to {ann_index_path} using pickle w/ protocol=pickle.HIGHEST_PROTOCOL")
    # with open(ann_index_path, "wb") as f:
    #     pickle.dump(ann_index, f, protocol=pickle.HIGHEST_PROTOCOL)


    # 2. Use Compression While Saving
    #Using compression reduces memory usage during serialization and the size of the saved file:
    # import gzip
    # Save with gzip compression
    # print(f"Attempting to save the ANN index to {ann_index_path} using gziped compressed pickle w/ protocol=pickle.HIGHEST_PROTOCOL")
    # with gzip.open(ann_index_path, "wb") as f:
        # pickle.dump(ann_index, ann_index_path, protocol=pickle.HIGHEST_PROTOCOL)

    # Convert back to SCR matrix and save - doesn't work
    # index_scr = scipy.sparse.csr_matrix(ann_index)
    # scipy.sparse.save_npz(  
    #     ann_index_path,
    #     index_scr,
    # )

    # This doesn't work :()
    # scipy.sparse.save_npz(  
    #     ann_index_path,
    #     # index.astype(numpy.float32),
    #     ann_index,
    # )

    saving_end = datetime.datetime.now()
    saving_time = saving_end - saving_start
    print(f"Saving the tfid vectorizer took {saving_time.total_seconds()} seconds")

    # return concept_aliases, tfidf_vectorizer, index

    # print(f"Fitting ANN index on {len(concept_aliases)} aliases using nmslib(takes 2 hours)")
    # fitting_ann_index_start = datetime.datetime.now()
    # ann_index = nmslib.init(
    #     method="hnsw",
    #     space="cosinesimil_sparse",
    #     data_type=nmslib.DataType.SPARSE_VECTOR,
    # )
    # ann_index.addDataPointBatch(concept_alias_tfidfs)
    # ann_index.createIndex(index_params, print_progress=True)
    # ann_index.saveIndex(ann_index_path)
    # fitting_ann_index_end = datetime.datetime.now()
    # fitting_ann_index_time = fitting_ann_index_end - fitting_ann_index_start
    # print(f"Fitting ann index took {fitting_ann_index_time.total_seconds()} seconds")
    
    print(f"Script finished at {datetime.datetime.now()}")
    return concept_aliases, tfidf_vectorizer, ann_searcher
