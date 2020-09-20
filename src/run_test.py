import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import transformers

import config
from model import TweetReconModel
from engine import test_fn, save_result
from dataset import TestFeatures

def run_test():
    data_dir = config.DATA_DIR
    bert_path = config.MODEL_OUTPUT_DIR
    tokenizer_path = f"{config.MODEL_OUTPUT_DIR}/vocab.txt"
    max_sequence_len = config.MAX_SQUENCE_LEN
    batch_size = config.TEST_BATCH_SIZE
    tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
    tFeatures = TestFeatures()
    test_features = tFeatures.get_test_features(data_dir, max_sequence_len=max_sequence_len, tokenizer=tokenizer)
    save_path = f"{config.SAVE_DIR}/result.csv"
    ids = torch.tensor([tf["id"] for tf in test_features])
    input_ids = torch.tensor([tf["input_ids"] for tf in test_features])
    token_type_ids = torch.tensor([tf["token_type_ids"] for tf in test_features])
    attention_mask = torch.tensor([tf["attention_mask"] for tf in test_features])

    test_dataset = TensorDataset(ids, input_ids, token_type_ids, attention_mask)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TweetReconModel(bert_path)
    model.to(device)
    result = []
    result = test_fn(test_dataloader=test_dataloader, model=model, device=device, result=result)

    save_result(result, save_path)

if __name__ == "__main__":
    run_test()