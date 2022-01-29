import csv
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
import time
from .test import test_classify


def train(
    model,
    data_loader,
    test_loader,
    numEpochs,
    device,
    criterion,
    optimizer,
    lr_scheduler,
    task="Classification",
):
    model.train()

    min_loss = 100

    for epoch in range(numEpochs):
        avg_loss = 0.0

        start_time = time.time()

        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            # outputs = model(feats)[1]
            outputs = model(feats)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 1000 == 990:
                print(
                    "Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}".format(
                        epoch + 1, batch_num + 1, avg_loss / 1000
                    )
                )
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == "Classification":
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)

            end_time = time.time()

            print("Epoch:", epoch)
            print(
                "Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}".format(
                    train_loss, train_acc, val_loss, val_acc
                )
            )

            print("Time:", end_time - start_time)

            # Save the order with min loss
            if val_loss < min_loss:
                torch.save(model.state_dict(), "resnet_v3.pth")
                min_loss = val_loss
                print("Model saved at Val Loss:", val_loss)
                print("=" * 40)

            lr_scheduler.step(val_loss)

        else:
            test_verify(model, test_loader)


def test_verify(model, test_loader):
    raise NotImplementedError