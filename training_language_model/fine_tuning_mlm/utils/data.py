from torch.utils.data import Dataset


# Group all text in chunk of size chunk_size
def group_texts(examples, chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

# it accepts an HF tokenized dataframe as input
# and converts it in a list of Dicts good for HF data collator
def refactor_dataset(dataset):
    refactored_set = [{"input_ids": dataset["input_ids"][i],
            "attention_mask": dataset["attention_mask"][i],
            "labels": dataset["labels"][i]   #not necessary for MLM, but is used for CLM, so we keep it
            } for i in range(len(dataset["input_ids"]))]
    
    return refactored_set

# Custom Dataset
# doesn't tokenize on the fly. We have to process all text toghether before feeding the models to create chunks. 
# inside tokenization makes sense for sentiment
class MLMData(Dataset):
    def __init__(self, encodings, data_collator):
        # encoding tokenized and chunked toghether as input for CLM/MLM
        self.encoding = encodings
        self.data_collator = data_collator

    def __len__(self):
        return len(self.encoding)

    def __getitem__(self, index):
        # it returns a single row of the list of dictionaries converted as tensor objects.
        # Keys are:
        #   input_ids
        #   attention_mask
        #   labels
        collated_data = self.data_collator([self.encoding[index]])
        # data collator need a list to be passed as input and return a dict with values as list of lists
        # since we feed only one row the list of lists contains only a lsit
        # we need to squeeze it taking only the inside list, otherwise the model fails since accepts as inputs tensor.shape(batch, max_len)
        item = {key: val[0] for key, val in collated_data.items()}
        #item = {key: val for key, val in self.encoding[index].items()}
        return item