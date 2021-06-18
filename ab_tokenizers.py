from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class TokenNumberizer:
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    def __init__(self, token_lookup={}):
        self.lookup = token_lookup
        self.tokens = TokenNumberizer.dict_sort(self.lookup)
        self.pretokenizer = Tokenizer(BPE())
        self.pretokenizer.pre_tokenizer = Whitespace()
    
    @staticmethod
    def convert_to_base(decimal_number: int, base: int, tokens) -> int:
        if decimal_number == 0:
            return []
        return tokens[decimal_number % base] + TokenNumberizer.convert_to_base(decimal_number // base, base, tokens)
    
    @staticmethod
    def base_to_dec(lst, base: int, tokens) -> int:
        if lst == []:
            return 0
        return tokens.index(lst[0]) + TokenNumberizer.base_to_dec(lst[1:], base, tokens) * base
    
    @staticmethod
    def dict_sort(dictionary: dict) -> list:
        """Takes in a dictionary with integer values and outputs a list of the keys sorted by their associated values in descending order."""
        return list(reversed(sorted(dictionary, key=dictionary.__getitem__)))
    
    def adapt(self, strings):
        self.pretokenizer.train_from_iterator(strings, trainer=TokenNumberizer.trainer)
        for string in strings:
            for token in self.pretokenizer.encode(string):
                self.lookup[token] = self.lookup.get(token, 0) + 1
        self.tokens = TokenNumberizer.dict_sort(self.lookup)
    
    def stoi(self, string):
        return TokenNumberizer.base_to_dec(self.pretokenizer.encode(string), len(self.tokens), self.tokens)
    
    def itos(self, number):
        return self.pretokenizer.decode(TokenNumberizer.convert_to_base(number, len(self.tokens), self.tokens))