from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(),
     transforms.ToTensor(), normalize])
val_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
