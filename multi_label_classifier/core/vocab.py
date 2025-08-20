"""
Vocabulary management for weather chart labels.
Handles token indexing, frequency counting, and special tokens.
"""

import os
import logging
from typing import Dict, List
import pandas as pd
from ...inspector import GalleryInspector, GalleryStats
from ...constants import GALLERY_DIR

logger = logging.getLogger(__name__)

__all__ = ["vocabulary"]

class Vocabulary:
    """
    Singleton class for managing weather chart label vocabulary.
    Includes token indexing, frequency-based filtering, and special tokens.
    """
    _instance = None  # singleton

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: str, min_frequency: int = 10, max_sequence_length: int = 5):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.base_dir:str = base_dir
        self.max_sequence_length: int = max_sequence_length
        self.min_frequency: int = min_frequency
        inspector = GalleryInspector(base_dir)
        self.stats: GalleryStats = inspector.inspect()
        counter: pd.DataFrame = self.count_corpus()
        self._token_freqs = counter.values.tolist()
        self.idx2token: List[str] = ['<unk>', '<bos>', '<eos>']
        self.token2idx: Dict[str, int] = {token: idx for idx, token in enumerate(self.idx2token)}
        for token, freq in self._token_freqs:
            if freq < self.min_frequency:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1
        logger.info("Vocabulary built with %d tokens", len(self.idx2token))

    def count_corpus(self) -> pd.DataFrame:
        """
        Count frequency of each label in the dataset.
        Returns a DataFrame with label frequencies sorted in descending order.
        """
        frequency : Dict[str, int] = {}
        for kinds, kind_stats in self.stats.kinds.items():
            kind_list = kinds.split("A")
            for kind in kind_list:
                frequency[kind] = frequency.get(kind, 0) + kind_stats.image_count
        corpus = pd.DataFrame(frequency.items(), columns=["kind", "frequency"]).sort_values(by="frequency", ascending=False)
        corpus.to_csv(os.path.join(self.base_dir, "corpus.csv"), index=False)
        logger.info("Corpus counted")
        return corpus

    def __len__(self) -> int:
        return len(self.idx2token)

    def embedding(self, tags: List[str], add_boseos: bool = True) -> List[int]:
        """
        Convert label tags to token indices.

        Args:
            tags: List of label tags to convert
            add_boseos: Whether to add BOS/EOS tokens

        Returns:
            List of token indices
        """
        indices: List[int] = [self.token2idx.get(tag, self.unk) for tag in tags]
        if add_boseos:
            indices = [self.bos] + indices + [self.eos]
        return indices

    def detokenize(self, indices: List[int], keep_bos_eos: bool = False) -> List[str]:
        """
        Convert token indices back to label tags.

        Args:
            indices: List of token indices to convert
            keep_bos_eos: Whether to keep BOS/EOS tokens in output

        Returns:
            List of label tags
        """
        tokens = []
        for idx in indices:
            if idx < 0 or idx >= len(self.idx2token):
                tokens.append("<unk>")
            else:
                tokens.append(self.idx2token[idx])

        if not keep_bos_eos:
            if tokens and tokens[0] == "<bos>":
                tokens = tokens[1:]
            if tokens and tokens[-1] == "<eos>":
                tokens = tokens[:-1]

        return tokens

    @property
    def unk(self):
        """Index of the unknown token (<unk>)"""
        return 0

    @property
    def bos(self):
        """Index of the beginning-of-sequence token (<bos>)"""
        return 1

    @property
    def eos(self):
        """Index of the end-of-sequence token (<eos>)"""
        return 2

vocabulary = Vocabulary(GALLERY_DIR)
