from torchtext.datasets import WikiText2, WikiText103
from torchtext.vocab import build_vocab_from_iterator
import torch

from warnings import warn


class WikiVocabulary():
    def __init__(self, tokenizer, vocab_path=None, save_path=None):
        self.vocab_path = vocab_path
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.vocab = None

    def _create_vocab(self):
        # create vocabulary tokenizing raw text
        train_iter = WikiText103(split='train')
        vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        if self.save_path:
            torch.save(vocab, f"{self.save_path}/vocab.pth")

        return vocab

    def _load_vocab(self):
        # load vocabulary
        return torch.load(f"{self.vocab_path}/vocab.pth")

    def get_vocab(self):
        try:
            self.vocab = self._load_vocab(self.vocab_path)
        except:
            if self.vocab_path is not None:
                warn(f"No vocabulary found at path {self.vocab_path}, creating one...")
            # generate vocab and cache file path to later use
            self.vocab = self._create_vocab()
            self.vocab_path = self.save_path

        return self.vocab



