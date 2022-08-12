import collections
import numpy as np

class Postprocessor :

    def __init__(self, args, raw_dataset, tokenizer) :
        self.args = args
        self.tokenizer = tokenizer
        self.raw_dataset = raw_dataset
        self.encoded_dataset = self._build(raw_dataset)
        
    def _build(self, dataset) :
        questions = [d["question"] for d in dataset]
        contexts = [d["context"] for d in dataset]

        encoded_dataset = self.tokenizer(questions, 
            contexts,
            truncation="only_second",
            max_length=self.args.max_seq_length,
            stride=self.args.stride,
            return_token_type_ids=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        return encoded_dataset

    def predict(self, predictionss) :

        prediction_texts = []

        guid2context = {d["id"] : d["context"] for d in self.raw_dataset}
        guids = list(guid2context.keys())

        start_logits = predictionss["start_logits"]
        end_logits = predictionss["end_logits"]
        impossible_logits = predictionss["impossible_logits"]

        guid_mapping = collections.defaultdict(list)
        for i, m in enumerate(self.encoded_dataset["overflow_to_sample_mapping"]) :
            guid = guids[m]
            guid_mapping[guid].append(i)

        for guid in guid_mapping :
            indices = guid_mapping[guid]

            guid_text = ""
            guid_no_answer_logit = -np.inf
            guid_score = -np.inf
            for i in indices :
                start_logit = start_logits[i]
                end_logit = end_logits[i]
                impossible_logit = impossible_logits[i]

                if guid_no_answer_logit < impossible_logit :
                    guid_no_answer_logit = impossible_logit

                impossible_flag = np.where(impossible_logit > 0.0, True, False)
                if not impossible_flag :

                    sequence_ids = self.encoded_dataset.sequence_ids(i)
                    offset_mapping = self.encoded_dataset["offset_mapping"][i]

                    context_start_id = 0
                    while sequence_ids[context_start_id] != 1 :
                        context_start_id += 1

                    context_end_id = len(sequence_ids) - 1
                    while sequence_ids[context_end_id] != 1 :
                        context_end_id -= 1

                    start_id = np.argmax(start_logit)
                    end_id = np.argmax(end_logit)

                    if start_id == 0 and end_id == 0 :
                        continue

                    if context_start_id <= start_id and end_id <= context_end_id :
                        score = start_logit[start_id] + end_logit[end_id]

                        if guid_score < score :
                            score = score
                            char_start_id = offset_mapping[start_id][0]
                            char_end_id = offset_mapping[end_id][1]

                            guid_text = guid2context[guid][char_start_id:char_end_id]

            prediction_texts.append({"prediction_text" : guid_text, "id" : guid, "no_answer_probability" : guid_no_answer_logit})
        
        return prediction_texts
