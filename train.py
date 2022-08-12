import os
import copy
import torch
import random
import numpy as np
import argparse
import wandb
import transformers
from tqdm import tqdm
from dotenv import load_dotenv

from utils.loader import Loader
from utils.encoder import Encoder
from utils.postprocessor import Postprocessor

from dataset.collator import PaddingCollator
from dataset.dataset import QuestionAnsweringDataset
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer
from models.metrics import SquadMetric
from models.scheduler import LinearWarmupScheduler
from models.model import RobertaForQuestionAnswering

def train(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    # -- Loading Datasets
    print("\nLoading Datasets")
    loader = Loader(data_dir=args.data_dir)
    train_dataset = loader.get_dataset(file_name=args.train_filename)
    validation_dataset = loader.get_dataset(file_name=args.dev_filename)

    # -- Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # -- Postprocessor
    reference_dataset = copy.deepcopy(validation_dataset)
    postprocessor = Postprocessor(args=args, tokenizer=tokenizer, raw_dataset=reference_dataset)

    # -- Validation Labels
    validation_labels = [
        {
            "id" : d["id"], 
            "answers" : {"answer_start" : d["answer_start"], "text" : d["answer_text"]}
        } 
        for d in validation_dataset
    ]

    # -- Encoding Datasets
    print("\nEncoding Datasets")
    encoder = Encoder(args, tokenizer=tokenizer)
    train_dataset = encoder(train_dataset)
    validation_dataset = encoder(validation_dataset)

    # -- Collator
    data_collator = PaddingCollator(max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    train_dataset = QuestionAnsweringDataset(dataset=train_dataset)
    print("\nThe number of train dataset : %d" %len(train_dataset))
    validation_dataset = QuestionAnsweringDataset(dataset=validation_dataset)
    print("The number of validation dataset : %d" %len(validation_dataset))

    # -- Dataloader
    train_dataloader = DataLoader(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )
    train_data_iterator = iter(train_dataloader)

    validation_dataloder = DataLoader(validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )

    # -- Config & Model
    config = AutoConfig.from_pretrained(args.PLM)

    model = RobertaForQuestionAnswering.from_pretrained(args.PLM, config=config)
    model.to(device)

    # -- Optimizer & Scheduler
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LinearWarmupScheduler(optimizer=optimizer, 
        total_steps=total_steps, 
        warmup_steps=warmup_steps
    )

    # -- Loss
    loss_ce = torch.nn.CrossEntropyLoss().to(device)
    loss_bce = torch.nn.BCEWithLogitsLoss().to(device)

    # -- Metric
    metrics = SquadMetric()

    # -- Wandb
    load_dotenv(dotenv_path="wandb.env")
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    name = f"EP:{args.epochs}_BS:{args.batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}"
    wandb.init(
        entity="sangha0411",
        project="klue-mrc",
        group=args.PLM,
        name=name
    )

    training_args = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio}
    wandb.config.update(training_args)

    # -- Training
    train_position_loss, train_impossible_loss = 0.0, 0.0
    for step in tqdm(range(total_steps)) :
        data = next(train_data_iterator)
        optimizer.zero_grad()
        # preparing inputs
        input_ids, attention_mask = data["input_ids"], data["attention_mask"]
        start_positions, end_positions, is_impossible = data["start_positions"], data["end_positions"], data["is_impossible"]
        
        # setting deivce
        input_ids = input_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        start_positions = start_positions.long().to(device)
        end_positions = end_positions.long().to(device)
        is_impossible = is_impossible.float().to(device)

        # forward model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits, end_logits, impossible_logits = outputs["start_logits"], outputs["end_logits"], outputs["impossible_logits"]
        
        # calculating loss
        start_loss = loss_ce(start_logits, start_positions)
        end_loss = loss_ce(end_logits, end_positions)
        position_loss = (start_loss + end_loss) / 2
        train_position_loss += position_loss.item()

        impossible_loss = loss_bce(impossible_logits, is_impossible)
        train_impossible_loss += impossible_loss.item()
        loss = position_loss + impossible_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        if step > 0 and step % args.logging_steps == 0 :
            train_position_loss /= args.logging_steps
            train_impossible_loss /= args.logging_steps
            info = {"train/learning_rate" : optimizer.param_groups[0]["lr"], 
                "train/position_loss" : round(train_position_loss, 4), 
                "train/impossible_loss" : round(train_impossible_loss, 4),
                "train/step" : step
            }
            print(info)
            wandb.log(info)

            train_position_loss = 0.0
            train_impossible_loss = 0.0

        # -- Validation
        if step > 0 and step % args.save_steps == 0 :
            with torch.no_grad() :
                model.eval()
                start_logit_vectors, end_logit_vectors, impossible_logit_vectors = [], [], []
                validation_position_loss, validation_impossible_loss = 0.0, 0.0
                print("Evaluation Model at step %d" %step)

                # evaluating model
                for validation_data in tqdm(validation_dataloder) :
                    # preparing inputs                        
                    input_ids, attention_mask = validation_data["input_ids"], validation_data["attention_mask"]
                    start_positions, end_positions, is_impossible = validation_data["start_positions"], validation_data["end_positions"], validation_data["is_impossible"]
                    
                    # setting deivce
                    input_ids = input_ids.long().to(device)
                    attention_mask = attention_mask.long().to(device)
                    start_positions = start_positions.long().to(device)
                    end_positions = end_positions.long().to(device)
                    is_impossible = is_impossible.float().to(device)

                    # forward model
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    start_logits, end_logits, impossible_logits = outputs["start_logits"], outputs["end_logits"], outputs["impossible_logits"]
                    
                    start_logit_vectors.extend([logit for logit in start_logits.detach().cpu().numpy()])
                    end_logit_vectors.extend([logit for logit in end_logits.detach().cpu().numpy()])
                    impossible_logit_vectors.extend([logit for logit in impossible_logits.detach().cpu().numpy()])

                    # calculating loss
                    start_loss = loss_ce(start_logits, start_positions)
                    end_loss = loss_ce(end_logits, end_positions)
                    validation_position_loss += (start_loss + end_loss) / 2
                    validation_impossible_loss += loss_bce(impossible_logits, is_impossible)

                # validation loss
                validation_position_loss /= len(validation_dataloder)
                validation_impossible_loss /= len(validation_dataloder)

                # postprocess validation logits
                prediction_logit_vectors = {"start_logits" : start_logit_vectors, "end_logits" : end_logit_vectors, "impossible_logits" : impossible_logit_vectors}
                validation_predictions = postprocessor.predict(prediction_logit_vectors)

                # validation metircs
                validation_metrics = metrics.compute_metrics(validation_predictions, validation_labels)
                validation_info = {"eval/" + k : v for k, v in validation_metrics.items()}
                validation_info["eval/position_loss"] = round(validation_position_loss.item(), 4)
                validation_info["eval/impossible_loss"] = round(validation_impossible_loss.item(), 4)
                print(validation_info)
                wandb.log(validation_info)

                # saving model
                checkpoint_path = os.path.join(args.output_dir, "checkpoint-%d" %step)
                if not os.path.exists(checkpoint_path) :
                    os.makedirs(checkpoint_path)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                model.train()

    # -- Finishing wandb
    wandb.finish()

    # -- Saving Model & Tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--epochs", type=int, default=5
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500
    )
    parser.add_argument(
        "--save_steps", type=int, default=2000
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3
    )
    parser.add_argument(
        "--PLM", type=str, default="klue/roberta-base"
    )
    parser.add_argument(
        "--data_dir", type=str, default="klue-mrc-v1.1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
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
        "--train_filename",
        default="klue-mrc-v1.1_train.json",
        type=str,
        help="Name of the train file (default: klue-re-v1.1_train.json",
    )
    parser.add_argument(
        "--dev_filename",
        default="klue-mrc-v1.1_dev.json",
        type=str,
        help="Name of the train file (default: klue-re-v1.1_dev.json",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
    )

    args = parser.parse_args()
    train(args)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
