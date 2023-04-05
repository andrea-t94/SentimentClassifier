from torch.utils.data import Dataset
import torch

# Custom Dataset
# Tokenize, pad and truncate on the fly
# Since it's a binary classificator, map labels [0,4] --> [0,1]
class SentimentData(Dataset):
    def __init__(self, text, sentiment, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text.to_list()
        self.targets = sentiment.to_list()
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
             # maps [0,4] --> [0,1]
            'labels': (torch.tensor(self.targets[index]) > torch.tensor(2)).float()
        }