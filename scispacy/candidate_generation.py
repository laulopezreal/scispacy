import logging
import os
from typing import Optional, List, Dict, Tuple, NamedTuple, Type
import json
import datetime
from collections import defaultdict
from sklearn.preprocessing import normalize

import scipy
import numpy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# import nmslib
# from nmslib.dist import FloatIndex
from pynndescent import NNDescent


from scispacy.util import scipy_supports_sparse_float16
from scispacy.file_cache import cached_path
from scispacy.linking_utils import (
    KnowledgeBase,
    UmlsKnowledgeBase,
    Mesh,
    GeneOntology,
    RxNorm,
    HumanPhenotypeOntology,
)

logger = logging.getLogger(__name__)
output_path = "./output_test"
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


UmlsLinkerPaths = LinkerPaths(
    # ann_index="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/nmslib_index.bin",  # noqa
    ann_index=f"{output_path}/ann_index.joblib",
    tfidf_vectorizer="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectorizer.joblib",  # noqa
    tfidf_vectors="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectors_sparse.npz",  # noqa
    concept_aliases_list="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/concept_aliases.json",  # noqa
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
    # ef_search: int = 200,
) -> NNDescent:
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
    # Load the ANN index using joblib
    ann_index = joblib.load(cached_path(linker_paths.ann_index))

    return ann_index

    # # Load the TF-IDF vectors
    # concept_alias_tfidfs = scipy.sparse.load_npz(
    #     cached_path(linker_paths.tfidf_vectors)
    # ).astype(numpy.float32)

    # Initialize the nmslib index
    # ann_index = nmslib.init(
    #     method="hnsw",
    #     space="cosinesimil_sparse",
    #     data_type=nmslib.DataType.SPARSE_VECTOR,
    # )

    # if linker_paths.ann_index:
    #     try:
    #         ann_index = joblib.load(cached_path(linker_paths.ann_index))
    #     except Exception as e:
    #         print("Failed to load ann index from disk. Rebuilding.")
    #         ann_index = NNDescent(
    #             concept_alias_tfidfs,
    #             n_neighbors=n_neighbors,
    #             metric='cosine',
    #             n_jobs=n_jobs,
    #             sparse=True,
    #         )
    # # Optional: Save the index to disk
    # joblib.dump(ann_index, 'path_to_save_index')

    # # # Add data points to the index
    # # ann_index.addDataPointBatch(concept_alias_tfidfs)

    # # # Load the pre-built index from disk
    # # ann_index.loadIndex(cached_path(linker_paths.ann_index))

    # # # Set query time parameters
    # # query_time_params = {"efSearch": ef_search}
    # # ann_index.setQueryTimeParams(query_time_params)

    # return ann_index


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
        ann_index: Optional[NNDescent] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
        ann_concept_aliases_list: Optional[List[str]] = None,
        kb: Optional[KnowledgeBase] = None,
        verbose: bool = False,
        ef_search: int = 200,
        name: Optional[str] = None,
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

        # Load the ANN index
        try:
            self.ann_index = ann_index or joblib.load(
                cached_path(linker_paths.ann_index)
            )
        except FileNotFoundError:
            print(
                f"Could not find ANN index at {linker_paths.ann_index}. "
                            "Creating a new index."
                            )
            ann_concept_aliases_list, tfidf_vectorizer, ann_index = create_tfidf_ann_index(
                output_path, kb
            )

        self.vectorizer = tfidf_vectorizer or joblib.load(
            cached_path(linker_paths.tfidf_vectorizer)
        )
        self.ann_concept_aliases_list = ann_concept_aliases_list or json.load(
            open(cached_path(linker_paths.concept_aliases_list))
        )

        self.kb = kb or DEFAULT_KNOWLEDGE_BASES[name]()
        self.verbose = verbose

        # TODO(Mark): Remove in scispacy v1.0.
        self.umls = self.kb

    # Remove the nmslib_knn_with_zero_vectors method, as PyNNDescent can handle zero vectors.
    # def nmslib_knn_with_zero_vectors(
    #     self, vectors: numpy.ndarray, k: int
    # ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    #     """
    #     ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
    #     This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
    #     - remove empty vectors from `vectors`.
    #     - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
    #     a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
    #     - extend the list `neighbors` with `None`s in place of empty vectors.
    #     - return the extended list of neighbors and distances.
    #     """
    #     empty_vectors_boolean_flags = numpy.array(vectors.sum(axis=1) != 0).reshape(-1)
    #     empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
    #     if self.verbose:
    #         print(f"Number of empty vectors: {empty_vectors_count}")

    #     # init extended_neighbors with a list of Nones
    #     extended_neighbors = numpy.empty(
    #         (len(empty_vectors_boolean_flags),), dtype=object
    #     )
    #     extended_distances = numpy.empty(
    #         (len(empty_vectors_boolean_flags),), dtype=object
    #     )

    #     if vectors.shape[0] - empty_vectors_count == 0:
    #         return extended_neighbors, extended_distances

    #     # remove empty vectors before calling `ann_index.knnQueryBatch`
    #     vectors = vectors[empty_vectors_boolean_flags]

    #     # call `knnQueryBatch` to get neighbors
    #     original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)

    #     neighbors, distances = zip(
    #         *[(x[0].tolist(), x[1].tolist()) for x in original_neighbours]
    #     )
    #     neighbors = list(neighbors)  # type: ignore
    #     distances = list(distances)  # type: ignore

    #     # neighbors need to be converted to an np.array of objects instead of ndarray of dimensions len(vectors)xk
    #     # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
    #     # returns an np.array of objects
    #     neighbors.append([])  # type: ignore
    #     distances.append([])  # type: ignore
    #     # interleave `neighbors` and Nones in `extended_neighbors`
    #     extended_neighbors[empty_vectors_boolean_flags] = numpy.array(
    #         neighbors, dtype=object
    #     )[:-1]
    #     extended_distances[empty_vectors_boolean_flags] = numpy.array(
    #         distances, dtype=object
    #     )[:-1]

    #     return extended_neighbors, extended_distances

    def __call__(
        self, mention_texts: List[str], k: int
    ) -> List[List[MentionCandidate]]:
        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")

        if not mention_texts:
            return []

        # Step 1: Vectorize and normalize the mention_texts
        tfidfs = self.vectorizer.transform(mention_texts)
        tfidfs = normalize(tfidfs, norm='l2', axis=1)

        # Step 2: Query the ANN index
        indices, distances = self.ann_index.query(tfidfs, k=k)

        # Step 3: Process the neighbors and distances
        batch_mention_candidates = []
        for neighbor_indices, neighbor_distances in zip(indices, distances):
            concept_to_mentions: Dict[str, List[str]] = defaultdict(list)
            concept_to_similarities: Dict[str, List[float]] = defaultdict(list)

            for neighbor_index, distance in zip(neighbor_indices, neighbor_distances):
                if neighbor_index == -1:
                    continue  # No neighbor found

                alias = self.ann_concept_aliases_list[neighbor_index]
                concepts_for_alias = self.kb.alias_to_cuis[alias]
                # Convert distance to similarity
                similarity = 1.0 - distance

                if similarity > 0.0:
                    for concept_id in concepts_for_alias:
                        concept_to_similarities[concept_id].append(similarity)
                        concept_to_mentions[concept_id].append(alias)

            # Create MentionCandidate instances
            mention_candidates = [
                MentionCandidate(concept_id, concept_to_mentions[concept_id], concept_to_similarities[concept_id])
                for concept_id in concept_to_similarities
                if concept_to_similarities[concept_id]  # Ensure similarities list is not empty
            ]

            # Sort candidates by maximum similarity (descending)
            mention_candidates.sort(key=lambda c: max(c.similarities), reverse=True)
            batch_mention_candidates.append(mention_candidates)

        return batch_mention_candidates



def create_tfidf_ann_index(
    out_path: str, kb: Optional[KnowledgeBase] = None
) -> Tuple[List[str], TfidfVectorizer, NNDescent]:
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
    os.makedirs(out_path, exist_ok=True)

    tfidf_vectorizer_path = f"{out_path}/tfidf_vectorizer.joblib"
    ann_index_path = f"{out_path}/nmslib_index.bin"
    tfidf_vectors_path = f"{out_path}/tfidf_vectors_sparse.npz"
    umls_concept_aliases_path = f"{out_path}/concept_aliases.json"

    kb = kb or UmlsKnowledgeBase()

    # nmslib hyperparameters (very important)
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
    print("Building TF-IDF vectorizer and vectors")
    concept_aliases = list(kb.alias_to_cuis.keys())

    # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
    # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
    # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
    # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408

    # 1. Build the TF-IDF vectorizer
    # This means that the tfidf vectors are stored in float32 format, but are loaded in float16 format
    print(f"Fitting tfidf vectorizer on {len(concept_aliases)} aliases")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=numpy.float32
    )
    start_time = datetime.datetime.now()
    concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
    # Normalize TF-IDF vectors
    concept_alias_tfidfs = normalize(concept_alias_tfidfs, norm='l2', axis=1)

    print(f"Saving tfidf vectorizer to {tfidf_vectorizer_path}")
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    scipy.sparse.save_npz(
        tfidf_vectors_path, concept_alias_tfidfs.astype(numpy.float32), compressed=True
    )

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"Fitting and saving vectorizer took {total_time.total_seconds()} seconds")

    # 4. Handle Empty TF-IDF Vectors
    # This code checks for and removes any empty TF-IDF vectors, 
    # which is important because they can cause issues during indexing.
    print("Finding empty (all zeros) tfidf vectors")
    empty_tfidfs_boolean_flags = numpy.array(
        concept_alias_tfidfs.sum(axis=1) != 0
    ).reshape(-1)

    number_of_non_empty_tfidfs = sum(empty_tfidfs_boolean_flags == False)  # noqa: E712
    total_number_of_tfidfs = numpy.size(concept_alias_tfidfs, 0)

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

    # 5. Save the Concept Aliases
    print(
        f"Saving list of concept ids and tfidfs vectors to {umls_concept_aliases_path} and {tfidf_vectors_path}"
    )
    json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))

    # 6. Initialize and Build the PyNNDescent Index
    # Replace the nmslib index creation with PyNNDescent:
    print(f"Fitting ANN index on {len(concept_aliases)} aliases (takes 2 hours with nmslib)")
    start_time = datetime.datetime.now()
    # Initialize the PyNNDescent index
    # Build the PyNNDescent index using 'cosine' metric
    ann_index = NNDescent(
        concept_alias_tfidfs,
        n_neighbors=10,        # Increase from 30 to 50
        n_trees=10,            # Increase from 20 to 30
        n_iters=10,            # Increase from 10 to 15
        max_candidates=10,    # Increase from 60 to 100
        delta=0.001,          # Decrease from 0.001 to 0.0001
        metric="cosine",
        n_jobs=1,
        random_state=42,
        verbose=True,
    )
    # ann_index = nmslib.init(
    #     method="hnsw",
    #     space="cosinesimil_sparse",
    #     data_type=nmslib.DataType.SPARSE_VECTOR,
    # )

    # ann_index.addDataPointBatch(concept_alias_tfidfs)
    # ann_index.createIndex(index_params, print_progress=True)
    # ann_index.saveIndex(ann_index_path)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Fitting ann index took {elapsed_time.total_seconds()} seconds")
    # 7. Save the Index
    # Since the PyNNDescent index is a Python object, you can save it using joblib.dump:

    ann_index_path = f"{out_path}/ann_index.joblib"
    joblib.dump(ann_index, ann_index_path)
    
    return concept_aliases, tfidf_vectorizer, ann_index


"""
Additional Considerations:
1. Parameter Tuning
n_neighbors: Adjust this parameter to balance between index construction time and search accuracy. A typical value is 15, but you might experiment with values like 10, 20, or 30.
n_jobs: Set to -1 to use all available CPU cores, or set to a specific number to limit CPU usage.

2. Compatibility with Other Parts of the Code
Ensure that wherever this function is called, the expected return types match the updated ones.
Update any code that loads the index to use joblib.load instead of nmslib methods.

3. Error Handling
You might want to add try-except blocks to handle potential errors during index creation or saving.
Ensure that the paths used for saving are valid and that you have write permissions.

4. Testing
After making these changes, test the index creation function to ensure it runs without errors.
Verify that the saved index can be loaded correctly and used for querying (as updated in previous steps).
"""