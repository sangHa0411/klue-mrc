from torch.utils.data import Dataset

class QuestionAnsweringDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]

        if "start_positions" in dataset :
            self.test_flag = False
            self.start_positions = dataset["start_positions"]
            self.end_positions = dataset["end_positions"]
            self.is_impossible = dataset["is_impossible"]
        else :
            self.test_flag = True
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        if self.test_flag :
            return {"input_ids" : input_ids, "attention_mask" : attention_mask}
        else :
            start_positions = self.start_positions[idx]
            end_positions = self.end_positions[idx]
            is_impossible = self.is_impossible[idx]
            return {"input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
                "start_positions" : start_positions, 
                "end_positions" : end_positions, 
                "is_impossible" : is_impossible
            }

