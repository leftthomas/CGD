import torchvision as tv


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        normalize = tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        val_augs = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)), 
            tv.transforms.ToTensor(), 
            normalize
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs
