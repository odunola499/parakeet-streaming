from typing import List, Union

import sentencepiece as spm


class ParakeetTokenizer:
    def __init__(self, tokenizer_path: str):
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required for ParakeetTokenizer")
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(tokenizer_path)

    def encode(self, texts: Union[str, List[str]]):
        return self.spm.encode(texts, out_type=int)

    def decode(self, texts: Union[List[int], List[List[int]]]) -> str:
        return self.spm.decode(texts)

    @property
    def pad_id(self) -> int:
        return self.spm.unk_id()

    @property
    def unk_id(self) -> int:
        return self.spm.unk_id()
