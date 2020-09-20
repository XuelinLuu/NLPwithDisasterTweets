import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import transformers
from transformers.file_utils import WEIGHTS_NAME
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import config
from model import TweetReconModel
from engine import train_fn, valid_fn
from dataset import TrainFeatures

def run_train():
    data_dir = config.DATA_DIR
    max_sequence_len = config.MAX_SQUENCE_LEN
    tokenizer_path = config.TOKENIZER_PATH
    batch_size = config.TRAIN_BATCH_SIZE
    epochs = config.TRAIN_EPOCHS
    bert_path = config.BERT_PATH
    learning_rate = config.LEARNING_RATE
    tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
    tFeatures = TrainFeatures()

    train_features = tFeatures.get_train_features(data_dir,max_sequence_len=max_sequence_len,tokenizer=tokenizer)
    input_ids = torch.tensor([tf["input_ids"] for tf in train_features])
    token_type_ids = torch.tensor([tf["token_type_ids"] for tf in train_features])
    attention_mask = torch.tensor([tf["attention_mask"] for tf in train_features])
    target = torch.tensor([tf["target"] for tf in train_features])
    target = F.one_hot(target, 2)
    target = torch.tensor(target.numpy(), dtype=torch.float)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, target)
    len_train, len_valid = int(0.9 * len(dataset)), (len(dataset) - int(0.9 * len(dataset)))
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train, len_valid])
    train_sampler = SequentialSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TweetReconModel(bert_path=bert_path)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) // batch_size * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        print(f"\n{epoch}/{epochs}\n-----------------------")
        train_fn(train_dataloader=train_dataloader,
                 model=model,
                 device=device,
                 optimizer=optimizer,
                 scheduler=scheduler
                 )
    

    model_save_path = os.path.join(config.MODEL_OUTPUT_DIR, WEIGHTS_NAME)
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_vocabulary(f"{config.MODEL_OUTPUT_DIR}/vocab.txt")
    print(f"\n\tvalid\t\n-----------------------")
    valid_fn(valid_dataloader, model, device)

if __name__ == "__main__":
    run_train()