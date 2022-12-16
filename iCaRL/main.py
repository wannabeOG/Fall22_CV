import argparse
import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from custom_folder import MyImageFolder
from data_loader import iCIFAR10, iCIFAR100
from model import iCaRLNet

def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()


# Hyper Parameters
total_classes = 10
num_classes = 10


transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Initialize CNN
K = 2000 # total number of exemplars
icarl = iCaRLNet(2048, 0)

parser = argparse.ArgumentParser(description='Test file')
parser.add_argument('--use_gpu', default=False, type=bool, help = 'Set the flag if you wish to use the GPU')
parser.add_argument('--batch_size', default=500, type=int, help = 'The batch size you want to use')
parser.add_argument('--num_freeze_layers', default=2, type=int, help = 'Number of layers you want to frozen in the feature extractor of the model')
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs you want to train the model on')
parser.add_argument('--init_lr', default=0.001, type=float, help='Initial learning rate for training the model')
parser.add_argument('--reg_lambda', default=0.01, type=float, help='Regularization parameter')

args = parser.parse_args()
use_gpu = args.use_gpu
batch_size = args.batch_size
no_of_layers = args.num_freeze_layers
num_epochs = args.num_epochs
lr = args.init_lr
reg_lambda = args.reg_lambda

dloaders_train = []
dloaders_test = []

train_sets = []
test_sets = []

dsets_train = []
dsets_test = []

num_classes = []

data_path = os.path.join(os.getcwd(), "DataSet")
data_dir = os.path.join(os.getcwd(), "DataSet")
data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),

	'test': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}


for tdir in sorted(os.listdir(data_dir)):

    #create the image folders objects
    if tdir==".DS_Store":
        continue
    path=os.path.join(data_dir, tdir, "train")
    tr_image_folder = MyImageFolder(os.path.join(data_dir, tdir, "train"), transform=data_transforms['train'])
    te_image_folder = MyImageFolder(os.path.join(data_dir, tdir, "test"), transform=data_transforms['test'])

    #get the dataloaders
    tr_dset_loaders = torch.utils.data.DataLoader(tr_image_folder, batch_size=batch_size, shuffle=False, num_workers=0)
    te_dset_loaders = torch.utils.data.DataLoader(te_image_folder, batch_size=batch_size, shuffle=False, num_workers=0)
    # get the sizes
    temp1 = len(tr_image_folder)
    temp2 = len(te_image_folder)
    dloaders_train.append(tr_dset_loaders)
    dloaders_test.append(te_dset_loaders)
    train_sets.append(tr_image_folder)
    test_sets.append(te_image_folder)

    # get the classes (THIS MIGHT NEED TO BE CORRECTED)
    num_classes.append(len(tr_image_folder.classes))

    # get the sizes array
    dsets_train.append(temp1)
    dsets_test.append(temp2)

no_of_tasks = len(dloaders_train)
a=0
for task in range(1, no_of_tasks+1):
    # Load Datasets
    # print("Loading training examples for classes", range(s, s+num_classes))
    # train_set = iCIFAR10(root='./data',
    #                      train=True,
    #                      classes=range(s,s+num_classes),
    #                      download=True,
    #                      transform=transform_test)


    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
    #                                            shuffle=True, num_workers=2)

    # test_set = iCIFAR10(root='./data',
    #                      train=False,
    #                      classes=range(num_classes),
    #                      download=True,
    #                      transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
    #                                            shuffle=True, num_workers=2)

    train_loader = dloaders_train[task - 1]
    test_loader = dloaders_test[task - 1]
    train_set=train_sets[task-1]
    test_set=test_sets[task-1]
    dset_size_train = dsets_train[task - 1]
    dset_size_test = dsets_test[task - 1]


    # Update representation via BackProp
    classes=[]
    for i in range(a, a+50):
        classes.append(i)
    a+=50
    icarl.update_representation(train_set, classes, train_loader)
    m = K / icarl.n_classes

    # Reduce exemplar sets for known classes
    icarl.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in range(icarl.n_known, icarl.n_classes):
        print("Constructing exemplar set for class-%d..." %(y),)
        images = []
        for i,(indices, images_temp, labels) in enumerate(train_loader):
            images = images_temp
            if i==y:
                break;
        icarl.construct_exemplar_set(images, m, data_transforms["test"])
        print("Done")

    for y, P_y in enumerate(icarl.exemplar_sets):
        print("Exemplar set for class-%d:" % (y), P_y.shape)
        #show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print("iCaRL classes: %d" % icarl.n_known)

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images)
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images)
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    # freeze_support()
    print('starting')