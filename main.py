from dataset.dataset import TrainDataset, TestDataset
from utils.transforms import get_transform
from utils.sampler import get_random_subset_sampler
from torch.utils.data import DataLoader
from routines.train import train
from routines.test import test_model
import pandas as pd
from config import numEpochs, criterion, lr_scheduler, device, optimizer
import torch
from config import config, model_conv
import os

image_dir = config["data"]["image_dir"]
data = config["data"]["train_data"]
test_data = config["data"]["test_data"]


def main():

    if config["mode"] == "train":

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

    if config["mode"] == "test":
        # Load model from previous training
        load_flag = config["model"]["load_trained_model"]
        if load_flag:
            model_conv.load_state_dict(
                torch.load(
                    os.path.join(
                        config["model"]["saved_models_folder"],
                        config["model"]["saved_model"],
                    )
                )
            )

        print(model_conv)

        testing_data = TestDataset(image_dir, test_data, transforms=get_transform(True))

        print(f"Length of test data: {len(testing_data)}")
        test_dataloader = torch.utils.data.DataLoader(
            testing_data,
            batch_size=len(testing_data),
            shuffle=False,
            num_workers=1,
        )
        print(f"Printing the test loader: {test_dataloader}")

        test_output = test_model(model_conv, test_dataloader, device="cpu")
        print(f"Length of the output: {len(test_output)}")
        print(test_output)

        test_probs = test_output[:, 1]
        print(f"Length of test probabilities: {len(test_probs)}")
        print("Printing the test probabilities")
        print(test_probs)

        # This code is to get the image names in a list
        # test_img_name = []
        test_image_names = pd.read_csv("data/sample_sub_v2.csv")["image"].tolist()

        print(len(test_image_names))
        pd.DataFrame({"image": test_image_names, "target": test_probs}).to_csv(
            config["data"]["output"], index=False
        )


if __name__ == "__main__":
    main()
