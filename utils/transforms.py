import torchvision


def get_transform(train):
    # transforms = []
    # # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(torchvision.transforms.Resize(size=(224, 224)))

    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    #     # transforms.append(torchvision.transforms.RandomRotation(5, resample=PIL.Image.BILINEAR))
    #     transforms.append(
    #         # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         torchvision.transforms.Normalize(
    #             (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #         )
    #     )

    # transforms.append(torchvision.transforms.ToTensor())

    # return torchvision.transforms.Compose(transforms)
    transforms_train = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(size=(800,600)),
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(degrees = (90,90)),
            # torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     mean=[0.8324, 0.5218, 0.5559], std=[0.1404, 0.1520, 0.1693]
            # ),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    transforms_test = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(size=(800,600)),
            torchvision.transforms.Resize(size=(224, 224)),
            # torchvision.transforms.RandomRotation(degrees = (90,90)),
            # torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     mean=[0.8324, 0.5218, 0.5559], std=[0.1404, 0.1520, 0.1693]
            # ),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    if train:
        return transforms_train
    else:
        return transforms_test
