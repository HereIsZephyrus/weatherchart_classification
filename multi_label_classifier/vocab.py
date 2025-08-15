"""
Vocabulary for the weather chart classification.
"""

import os
import logging
from typing import Dict, List
import pandas as pd
from inspector import GalleryInspector, GalleryStats
from ..constants import GALLERY_DIR

logger = logging.getLogger(__name__)

__all__ = ["vocabulary"]

class Vocabulary:
    """
    Vocabulary for the weather chart classification.
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
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx2token: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']
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
        Count the frequency of each kind
        """
        frequency : Dict[str, int] = {}
        for kinds, kind_stats in self.stats.kinds.items():
            kind_list = kinds.split("A")
            for kind in kind_list:
                frequency[kind] = frequency.get(kind, 0) + kind_stats.image_count
        corpus = pd.DataFrame(frequency.items(), columns=["kind", "frequency"])
        corpus.to_csv(os.path.join(self.base_dir, "corpus.csv"), index=False)
        logger.info("Corpus counted")
        return corpus

    def __len__(self) -> int:
        return len(self.idx2token)

    def embedding(self, tags: List[str], add_boseos: bool = True) -> List[int]:
        """
        Embedding the tags into indices
        """
        indices: List[int] = [self.token2idx.get(tag, self.unk) for tag in tags]
        if add_boseos:
            indices = [self.bos] + indices + [self.eos]
        return indices

    def detokenize(self, indices: List[int], keep_bos_eos: bool = False) -> List[str]:
        """
        Detokenize the indices
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
        """
        Unknown token
        """
        return 0

    @property
    def bos(self):
        """
        Begin of sentence token
        """
        return 1

    @property
    def eos(self):
        """
        End of sentence token
        """
        return 2

    @property
    def pad(self):
        """
        Padding token
        """
        return 3

vocabulary = Vocabulary(GALLERY_DIR)
