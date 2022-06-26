from dataset.dataset import TrainDataset, TestDataset
from utils.transforms import get_transform
from utils.sampler import get_random_subset_sampler
from torch.utils.data import DataLoader
from model.resnet import model_conv
from routines.train import train
from routines.test import test_model
import pandas as pd
from config import numEpochs, criterion, lr_scheduler, device, optimizer
import torch
from config import config
import os

image_dir = config["data"]["image_dir"]
data = config["data"]["train_data"]
test_data = config["data"]["test_data"]

# Load model from previous training
load_flag = config["model"]["load_trained_model"]
if load_flag:
    model_conv.load_state_dict(
        torch.load(
            os.path.join(
                config["model"]["saved_models_folder"], config["model"]["saved_model"]
            )
        )
    )

# print(model_conv)


def main():

    train_data = TrainDataset(image_dir, data, transforms=get_transform(True))

    train_sampler, val_sampler = get_random_subset_sampler(
        len_train_data=len(train_data),
        valid_size=config["hyperparams"]["cross_validation_split"],
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config["hyperparams"]["batch_size"],
        sampler=train_sampler,
        num_workers=config["hyperparams"]["num_workers"],
    )
    val_dataloader = DataLoader(
        train_data,
        batch_size=config["hyperparams"]["batch_size"],
        sampler=val_sampler,
        num_workers=config["hyperparams"]["num_workers"],
    )
    train(
        model=model_conv,
        data_loader=train_dataloader,
        test_loader=val_dataloader,
        numEpochs=numEpochs,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    testing_data = TestDataset(image_dir, test_data, transforms=get_transform(True))

    print(len(testing_data))
    test_dataloader = torch.utils.data.DataLoader(
        testing_data,
        batch_size=len(testing_data),
        shuffle=False,
        num_workers=config["hyperparams"]["num_workers"],
    )
    print(test_dataloader)

    test_output = test_model(model_conv, test_dataloader)
    print(len(test_output))

    test_probs = test_output[:, 1]
    print(len(test_probs))

    # This code is to get the image names in a list
    test_img_name = []
    for i in range(len(test_data)):
        test_img_name.append(test_data[i][0])

    print(len(test_img_name))
    pd.DataFrame({"image": test_img_name, "target": test_probs}).to_csv(
        "output.csv", index=False
    )

    return


if __name__ == "__main__":
    main()
