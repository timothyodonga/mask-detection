from dataset.dataset import TrainDataset, TestDataset
from utils.transforms import get_transform
from utils.sampler import get_random_subset_sampler
from torch.utils.data import DataLoader
from model.resnet import model_conv
from config import *
from routines.train import train
from routines.test import test_model
import sys
import pandas as pd

image_dir = sys.argv[1]
data = sys.argv[2]
test_data = sys.argv[3]
# data =

# Load model from previous training
load_flag = False
if load_flag == True:
    model_conv.load_state_dict(torch.load("resnet_v2.pth"))

print(model_conv)


def main():

    train_data = TrainDataset(image_dir, data, transforms=get_transform(True))

    train_sampler, val_sampler = get_random_subset_sampler(
        len_train_data=len(train_data), valid_size=0.2
    )

    train_dataloader = DataLoader(
        train_data, batch_size=64, sampler=train_sampler, num_workers=2
    )
    val_dataloader = DataLoader(
        train_data, batch_size=64, sampler=val_sampler, num_workers=2
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
        testing_data, batch_size=509, shuffle=False, num_workers=4
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
