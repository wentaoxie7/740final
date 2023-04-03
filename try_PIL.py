import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# load the CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)

# get an image from the dataset
img, label = cifar10_dataset[0]
print(type(img))

# define a transform to convert the tensor to a PIL Image
transform_to_pil = transforms.ToPILImage()

# convert the tensor to a PIL Image
pil_image = transform_to_pil(img)

# show the PIL Image
pil_image.show()
