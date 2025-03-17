from typing import Optional
import json
import datetime

import os
import bm25s
import faiss
import numpy as np
from tqdm import tqdm
from txtai.pipeline import Tokenizer
from scispacy.linking_utils import KnowledgeBase
from scispacy.candidate_generation import UmlsKnowledgeBase
from scispacy.util import get_embedding

# from transformers import AutoTokenizer, AutoModelForMaskedLM
import Stemmer

def create_tfidf_index(
    out_path: str, dense_index_path: str = "data/faiss_index.bin", kb: Optional[KnowledgeBase] = None, test_mode: bool = False, n_test: Optional[int]=1000,):
    """
    Build tfidf vectorizer and ann index.

    Parameters
    ----------
    out_path: str, required.
        The path where the various model pieces will be saved.
    kb : KnowledgeBase, optional.
        The kb items to generate the index and vectors for.

    """
    # Create a subfolder to save the linker artifacts
    #  Format datetime as YYYYmmddhhmm  
    date_subfolder = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_path = f"{out_path}{date_subfolder}"

    print(f"Creating subfolder to save the outputs at {output_path}")
    os.makedirs(output_path, exist_ok=True,)

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
    
    # Get concept aliases from Knowledge Base
    concept_aliases = list(kb.alias_to_cuis.keys())
    initial_n = len(concept_aliases)

    if test_mode:
        concept_aliases = concept_aliases[0:n_test] 
        print(f"Test mode enabled: reducing concept aliases from {initial_n} to {len(concept_aliases)} for testing")
    print(f"Saving concept aliases to {umls_concept_aliases_path}")
    json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))
    
    # if not os.path.exists(dense_index_path):
    #     print("Creating FAISS index...")
    #     # Generate embeddings for concepts
    #     concept_vectors = np.array([get_embedding(text) for text in tqdm(concept_aliases, leave=True)])
        
    #     # Create FAISS index (L2 distance)
    #     print("Creating FAISS index...")
    #     dimension = concept_vectors.shape[1]
    #     faiss_index = faiss.IndexFlatL2(dimension)
    #     faiss_index.add(concept_vectors)

    #     # Save FAISS index
    #     print(f"Saving FAISS index.")
    #     faiss.write_index(faiss_index, os.path.join(output_path, "faiss_index.bin"))
    # # np.save(os.path.join(output_path, "umls_concepts.npy"), np.array(concept_aliases))
    # else:
    #     print(f"✅ FAISS index already exists at {dense_index_path}")

    # print("✅ FAISS index created and saved.")

    # Test txtai tokenizer
    # print("Using txtai tokenizer with BM25S algorithm to create the index")
    # tokenizer = Tokenizer()
    # corpus_tokenized = [tokenizer(doc) for doc in concept_aliases]

    # bm25s tokenizer
    print("Using BM25S tokenizer to create the bm25s index")
    # print("Using BM25S tokenizer with Stemmer to create the index")
    # stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(
        # stemmer=stemmer, 
        # stopwords=None,
        )
    corpus_tokenized = tokenizer.tokenize(concept_aliases, return_as="string",)
    
    # SPLADE tokenizer and model: FAILS BC WE CANT ACCESS HUGGING FACE
    # tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
    # splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
    # splade_model.eval()
    # corpus_tokenized = [get_splade_sparse_vector(text, tokenizer, splade_model) for text in concept_aliases]

    k1=2.0
    b=1.2
    print(f"Parameters: b {b} k1 {k1}")
    
    # With Numba Just in Time (JIT)
    model = bm25s.BM25(backend="numba", method="lucene", k1=k1, b=b)

    # Without Numba
    # model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    # for x in concept_aliases[0:10]:
    #     print(x)

    model.index(corpus_tokenized, leave_progress=True)
    
    saving_start = datetime.datetime.now()
    model.save(output_path)
    saving_end = datetime.datetime.now()
    saving_time = saving_end - saving_start
    print(f"Saving the tfid vectorizer took {saving_time.total_seconds()} seconds")
    
    print(f"Script finished at {datetime.datetime.now()}")
    return concept_aliases, corpus_tokenized, model
