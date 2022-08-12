from tqdm import tqdm

class Encoder :

    def __init__(self, args, tokenizer) :
        self.tokenizer = tokenizer 
        self.max_seq_length = args.max_seq_length
        self.stride = args.stride

    def _encode(self, data) :
        dataset = []
        encoded_data = self.tokenizer(data["question"], 
            data["context"],
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.stride,
            return_token_type_ids=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        if "answer_text" not in data :
            for i in range(len(encoded_data["overflow_to_sample_mapping"])) :                
                input_ids = encoded_data["input_ids"][i]
                attention_mask = encoded_data["attention_mask"][i]

                result = {"input_ids" : input_ids, "attention_mask" : attention_mask}
                dataset.append(result) 
        else :
            flag = data["is_impossible"]
            if flag == True :
                for i in range(len(encoded_data["overflow_to_sample_mapping"])) :                
                    input_ids = encoded_data["input_ids"][i]
                    attention_mask = encoded_data["attention_mask"][i]

                    result = {"input_ids" : input_ids, 
                        "attention_mask" : attention_mask, 
                        "start_positions" : 0, 
                        "end_positions" : 0,
                        "is_impossible" : True,
                    }
                    dataset.append(result)
            else :    
                answer_text = data["answer_text"][0]

                answer_start = data["answer_start"][0]
                answer_end = data["answer_start"][0] + len(answer_text)
                
                for i in range(len(encoded_data["overflow_to_sample_mapping"])) :
                    input_ids = encoded_data["input_ids"][i]
                    attention_mask = encoded_data["attention_mask"][i]
                    offset_mapping = encoded_data["offset_mapping"][i]
                    sequence_ids = encoded_data.sequence_ids(i)

                    start_id = 0
                    while sequence_ids[start_id] != 1 :
                        start_id += 1

                    end_id = len(sequence_ids) - 1
                    while sequence_ids[end_id] != 1 :
                        end_id -= 1

                    token_positions = []
                    for j in range(start_id, end_id + 1) :
                        offset = offset_mapping[j]
                        start_char, end_char = offset

                        if answer_start <= start_char and end_char <= answer_end :
                            token_positions.append(j)
                            
                    if len(token_positions) > 0 :
                        start_position = min(token_positions)
                        end_position = max(token_positions)

                        result = {"input_ids" : input_ids, 
                            "attention_mask" : attention_mask, 
                            "start_positions" : start_position, 
                            "end_positions" : end_position,
                            "is_impossible" : False,
                        }
                    else :
                        result = {"input_ids" : input_ids, 
                            "attention_mask" : attention_mask, 
                            "start_positions" : 0, 
                            "end_positions" : 0,
                            "is_impossible" : True,
                        }
                    dataset.append(result)
        return dataset

    def __call__(self, dataset) :
        input_ids = []
        attention_mask = []
        start_positions = []
        end_positions = []
        is_impossibles = []

        for data in tqdm(dataset) :
            encoded_data = self._encode(data)
            for e in encoded_data :
                input_ids.append(e["input_ids"])
                attention_mask.append(e["attention_mask"])

                if "start_positions" in e :
                    start_positions.append(e["start_positions"])
                    end_positions.append(e["end_positions"])
                    is_impossibles.append(e["is_impossible"])

        if len(start_positions) > 0 :
            dataset = {"input_ids" : input_ids, 
                "attention_mask" : attention_mask,
                "start_positions" : start_positions,
                "end_positions" : end_positions,
                "is_impossible" : is_impossibles,
            }
        else :
            dataset = {"input_ids" : input_ids, "attention_mask" : attention_mask}
        return dataset