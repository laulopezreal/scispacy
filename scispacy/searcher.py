from typing import Optional
import json
import datetime

import os
import bm25s
from txtai.pipeline import Tokenizer
from scispacy.linking_utils import KnowledgeBase
from scispacy.candidate_generation import UmlsKnowledgeBase, get_splade_sparse_vector
from transformers import AutoTokenizer, AutoModelForMaskedLM
import Stemmer

def create_tfidf_index(
    out_path: str, tfidf_vectorizer_path: Optional[str] = None, kb: Optional[KnowledgeBase] = None, test_mode: bool = False, n_test: Optional[int]=1000,):
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

    # NMSLIB hyperparameters (very important)
    # guide: https://github.com/nmslib/nmslib/blob/master/manual/methods.md
    # Default values resulted in very low recall.

    # set to the maximum recommended value. Improves recall at the expense of longer indexing time.
    # We use the HNSW (Hierarchical Navigable Small World Graph) representation which is constructed
    # by consecutive insertion of elements in a random order by connecting them to M closest neighbours
    # from the previously inserted elements. These later become bridges between the network hubs that
    # improve overall graph connectivity. (bigger M -> higher recall, slower creation)
    # For more details see:  https://arxiv.org/pdf/1603.09320.pdf?
    # m_parameter = 100
    # `C` for Construction. Set to the maximum recommended value
    # Improves recall at the expense of longer indexing time
    # construction = 2000
    # num_threads = 60  # set based on the machine
    # index_params = {
    #     "M": m_parameter,
    #     "indexThreadQty": num_threads,
    #     "efConstruction": construction,
    #     "post": 0,
    # }
    
    # Get concept aliases from Knowledge b=Base
    concept_aliases = list(kb.alias_to_cuis.keys())
    initial_n = len(concept_aliases)

    if test_mode:
        concept_aliases = concept_aliases[0:n_test] 
        print(f"Test mode enabled: reducing concept aliases from {initial_n} to {len(concept_aliases)} for testing")
    print(f"Saving concept aliases to {umls_concept_aliases_path}")
    json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))

    # Test txtai tokenizer
    # print("Using txtai tokenizer with BM25S algorithm to create the index")
    # tokenizer = Tokenizer()
    # corpus_tokenized = [tokenizer(doc) for doc in concept_aliases]

    # bm25s tokenizer
    print("Using BM25S tokenizer with Stemmer to create the index")
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(
        # stemmer=stemmer, 
        stopwords=None,
        )
    corpus_tokenized = tokenizer.tokenize(concept_aliases, return_as="string",)
    
    # SPLADE tokenizer and model: FAILS BC WE CANT ACCESS HUGGING FACE
    # tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
    # splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
    # splade_model.eval()
    # corpus_tokenized = [get_splade_sparse_vector(text, tokenizer, splade_model) for text in concept_aliases]


    # print(type(corpus_tokenized))
    # for token in corpus_tokenized:
    #     if type(token) != str:
    #         print(type(token))
    #         print(token)
    # corpus_tokenized=[]
    # for alias in tqdm(concept_aliases,): 
    #     corpus_tokenized.append(tokenizer.tokenize(alias, leave_progress=False))

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
