
import os
import json

class Loader :

    def __init__(self, data_dir) :
        self.data_dir = data_dir

    def get_dataset(self, file_name) :
        path = os.path.join(self.data_dir, file_name)
        with open(path, 'r') as f :
            self.dataset = json.load(f)["data"]

        dataset = []
        for d in self.dataset :
            title = d["title"]
            paragraphs = d["paragraphs"][0]
            context = paragraphs["context"]

            qas = paragraphs["qas"]

            for qa in qas :
                question = qa["question"]
                guid = qa["guid"]

                if "answers" in qa :
                    answers = qa["answers"]
                    flag = qa["is_impossible"]

                    if flag == True :
                        data = {"question" : question,
                            "is_impossible" : flag,
                            "answer_start" : [], 
                            "answer_text" : [], 
                            "context" : context, 
                            "title" : title,
                            "id" : guid
                        }
                    else :
                        answer_start = [a["answer_start"] for a in answers]
                        answer_text = [a["text"] for a in answers]
                        data = {"question" : question,
                            "is_impossible" : flag,
                            "answer_start" : answer_start, 
                            "answer_text" : answer_text, 
                            "context" : context, 
                            "title" : title,
                            "id" : guid
                        }

                    dataset.append(data)
                else :
                    data = {"question" : question, "context" : context, "title" : title, "id" : guid}
                    dataset.append(data)
                
        return dataset
