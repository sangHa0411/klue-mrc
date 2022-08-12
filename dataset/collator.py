import torch

class PaddingCollator :
    def __init__(self, max_seq_length, tokenizer) :
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self, dataset) :
        input_ids = []
        attention_mask = []
        start_positions = []
        end_positions = []
        is_impossible = []

        for data in dataset :
            input_ids.append(data["input_ids"])
            attention_mask.append(data["attention_mask"])

            if "start_positions" in data :
                start_positions.append(data["start_positions"])
                end_positions.append(data["end_positions"])
                is_impossible.append(data["is_impossible"])

        batch_max_seq_length = max([len(x) for x in input_ids])
        batch_max_seq_length = min(self.max_seq_length, batch_max_seq_length)

        for i in range(len(dataset)) :
            seq_size = len(input_ids[i])

            if seq_size < batch_max_seq_length :
                padding_size = batch_max_seq_length - seq_size
                input_ids[i] = input_ids[i] + [self.tokenizer.pad_token_id] * padding_size
                attention_mask[i] = attention_mask[i] + [0] * padding_size
            else :
                input_ids[i] = input_ids[i][:batch_max_seq_length]
                attention_mask[i] = attention_mask[i][:batch_max_seq_length]

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int32)

        if len(start_positions) > 0 :
            start_positions = torch.tensor(start_positions, dtype=torch.int32)
            end_positions = torch.tensor(end_positions, dtype=torch.int32)
            is_impossible = torch.tensor(is_impossible, dtype=torch.int32)

            batch = {"input_ids" : input_ids, 
                "attention_mask" : attention_mask,
                "start_positions" : start_positions, 
                "end_positions" : end_positions, 
                "is_impossible" : is_impossible
            }
            return batch
        else :
            batch = {"input_ids" : input_ids, "attention_mask" : attention_mask}
            return batch