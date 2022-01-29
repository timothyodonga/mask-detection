import torchvision


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(torchvision.transforms.Resize(size=(224, 224)))

    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
        # transforms.append(torchvision.transforms.RandomRotation(5, resample=PIL.Image.BILINEAR))

    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    return torchvision.transforms.Compose(transforms)
