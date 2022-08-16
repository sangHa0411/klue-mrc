import os
import copy
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from utils.loader import Loader
from utils.encoder import Encoder
from utils.postprocessor import Postprocessor

from dataset.collator import PaddingCollator
from dataset.dataset import QuestionAnsweringDataset
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer
from models.model import RobertaForQuestionAnswering

def inference(args) :
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # -- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    print("\nLoading Datasets")
    loader = Loader(data_dir=args.data_dir)
    test_dataset = loader.get_dataset(file_name=args.test_filename)

    # -- Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # -- Postprocessor
    reference_dataset = copy.deepcopy(test_dataset)
    postprocessor = Postprocessor(args=args, tokenizer=tokenizer, raw_dataset=reference_dataset)

    # -- Encoding Datasets
    print("\nEncoding Datasets")
    encoder = Encoder(args, tokenizer=tokenizer)
    test_dataset = encoder(test_dataset)

    # -- Collator
    data_collator = PaddingCollator(max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    test_dataset = QuestionAnsweringDataset(dataset=test_dataset)
   
    # -- Dataloader
    test_dataloader = DataLoader(test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )

    # -- Config & Model
    config = AutoConfig.from_pretrained(args.PLM)

    model = RobertaForQuestionAnswering.from_pretrained(args.PLM, config=config)
    model.to(device)
    
    print("\nInferencing Datasets")
    with torch.no_grad() :
        model.eval()

        start_logit_vectors, end_logit_vectors, impossible_logit_vectors = [], [], []
        for data in tqdm(test_dataloader) :
            # preparing inputs                        
            input_ids, attention_mask = data["input_ids"], data["attention_mask"]

            # setting deivce
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            
             # forward model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits, end_logits, impossible_logits = outputs["start_logits"], outputs["end_logits"], outputs["impossible_logits"]
            
            start_logit_vectors.extend([logit for logit in start_logits.detach().cpu().numpy()])
            end_logit_vectors.extend([logit for logit in end_logits.detach().cpu().numpy()])
            impossible_logit_vectors.extend([logit for logit in impossible_logits.detach().cpu().numpy()])

        # postprocess validation logits
        prediction_logit_vectors = {"start_logits" : start_logit_vectors, "end_logits" : end_logit_vectors, "impossible_logits" : impossible_logit_vectors}
        predictions = postprocessor.predict(prediction_logit_vectors)

    # -- Writing output file
    print("\nWriting predictions to file")
    pred_ids = [pred["id"] for pred in predictions]
    pred_texts = [
        pred["prediction_text"] if pred["no_answer_probability"] > 0.0 else "" 
        for pred in predictions
    ]

    df = pd.DataFrame({"id" : pred_ids, "text" : pred_texts})
    df.to_csv(args.output_file, index=False)

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--PLM", type=str, default="klue/roberta-base"
    )
    parser.add_argument(
        "--data_dir", type=str, default="klue-mrc-v1.1"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.csv",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=510,
        help="maximum sequence length (default: 510)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="stride for overflow token mapping (default: 128)",
    )
    parser.add_argument(
        "--test_filename",
        default="klue-mrc-v1.1_test.json",
        type=str,
        help="Name of the test file (default: klue-re-v1.1_test.json",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
    )

    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()