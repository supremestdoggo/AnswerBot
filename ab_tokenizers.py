from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import *
from tokenizers.trainers import BpeTrainer
import tokenizers


# Required functions
def convert_to_base(number: int, base: int, tokens: list) -> list:
    if number == 0:
        return []
    return [tokens[number % base]] + TokenNumberizer.convert_to_base(number // base, base, tokens)

def base_to_dec(lst, base: int, tokens) -> int:
    if lst == []:
        return 0
    return tokens.index(lst[0]) + TokenNumberizer.base_to_dec(lst[1:], base, tokens) * base

def dict_sort(dictionary: dict) -> list:
    """Takes in a dictionary with integer values and outputs a list of the keys sorted by their associated values in descending order."""
    return list(reversed(sorted(dictionary, key=dictionary.__getitem__)))


class TokenNumberizer:
    """Simple class for token-based string-int conversion."""
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    def __init__(self, token_lookup={}):
        self.lookup = token_lookup
        self.tokens = TokenNumberizer.dict_sort(self.lookup)
        self.pretokenizer = Tokenizer(BPE())
        self.pretokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
    
    def adapt(self, strings):
        self.pretokenizer.train_from_iterator(strings, trainer=TokenNumberizer.trainer)
        for string in strings:
            for token in self.pretokenizer.encode(string).ids:
                self.lookup[token] = self.lookup.get(token, 0) + 1
        self.tokens = TokenNumberizer.dict_sort(self.lookup)
    
    def stoi(self, string: str):
        return TokenNumberizer.base_to_dec(self.pretokenizer.encode(string).ids, len(self.tokens), self.tokens)
    
    def itos(self, number: int):
        return self.pretokenizer.decode(TokenNumberizer.convert_to_base(number, len(self.tokens), self.tokens))