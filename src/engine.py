import os
import tqdm
import torch
import torch.nn as nn

import pandas as pd

def loss_fn(output, target, device):
    loss = nn.BCEWithLogitsLoss().to(device)
    return loss(output, target)

def train_fn(train_dataloader, model, device, optimizer, scheduler):
    model.train()
    all_loss = 0
    all_acc = 0
    tdm = tqdm.tqdm(train_dataloader, desc="Train Iter")
    for i, data in enumerate(tdm):
        input_ids, token_type_ids, attention_mask, targets = data[0], data[1], data[2], data[3]
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        o, t = torch.argmax(outputs, dim=-1), torch.argmax(targets, dim=-1)
        all_acc += torch.sum(o == t).detach().cpu().numpy().item()
        loss = loss_fn(outputs, targets, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        all_loss += loss.item()
        tdm.set_postfix(loss=loss.item(), avg_loss=(all_loss/(i+1)), avg_acc=(all_acc/(i+1)/32))

def valid_fn(valid_dataloader, model, device):
    model.eval()
    tdm = tqdm.tqdm(valid_dataloader)
    all_acc = 0
    all_loss = 0
    for i, data in enumerate(tdm):
        input_ids, token_type_ids, attention_mask, targets = data[0], data[1], data[2], data[3]
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        loss = loss_fn(output, targets, device)
        o, t = torch.argmax(output), torch.argmax(targets)
        all_acc += torch.sum(o == t).detach().cpu().numpy().item()
        all_loss += loss.item()
        tdm.set_postfix(avg_loss=(all_loss/(i+1)), avg_acc=(all_acc/((i+1)*32)))

def test_fn(test_dataloader, model, device, result):
    model.eval()
    tdm = tqdm.tqdm(test_dataloader, desc="Test Iter")
    for i, data in enumerate(tdm):
        id, input_ids, token_type_ids, attention_mask = data[0], data[1], data[2], data[3]
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

        output = torch.argmax(output, dim=-1)
        id = id.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        result += [[id[i], output[i]] for i in range(len(id))]
    return result

def save_result(result, save_path):
    columns = ["id", "target"]
    df = pd.DataFrame(data=result, columns=columns)
    df.to_csv(save_path, index=False)