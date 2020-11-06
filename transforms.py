import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(mode):
    transforms = []
    if mode == 'train':
        transforms.append(A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.25))
        transforms.append(A.CoarseDropout(min_holes=3, min_height=4, min_width=4, max_height=12, max_width=12, p=0.25))

    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms=transforms)
