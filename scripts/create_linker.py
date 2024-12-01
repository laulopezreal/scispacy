import argparse
import os
from datetime import datetime
from typing import Optional

from scispacy.candidate_generation import create_tfidf_ann_index
from scispacy.linking_utils import KnowledgeBase

DEFAULT_UMLS_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2023-04-23/umls_2022_ab_cat0129.jsonl"  # noqa
DEFAULT_UMLS_TYPES_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_semantic_type_tree.tsv"


def main(kb_path: Optional[str], output_path: str):
    os.makedirs(output_path, exist_ok=True)
    print(f"Running script at {datetime.now()}")
    kb = None
    if kb_path:
        kb = KnowledgeBase(kb_path)
    create_tfidf_ann_index(
        out_path=output_path,
        kb = kb,
        test_mode=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kb_path',
        help="Path to the KB file.",
        # required=True,
        default=None,
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output directory.",
        # required=True,
        default="output/"
    )

    args = parser.parse_args()
    main(args.kb_path, args.output_path)
