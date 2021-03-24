import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #, plot_confusion_matrix
import torch
import torchvision
#import torch.nn as nn
#import torch.nn.functional as f

from experiments.models import *
#from experiments.datasets import *

#from tensorboardX import SummaryWriter
#from ignite.engine import create_supervised_evaluator
#from ignite.metrics import Accuracy, Loss

#from pruning.pruning_algos import iterative_pruning
#from experiments.experiments import *
#from pruning.mask_networks import apply_prune_mask

#import os
#import argparse
#import random
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap='viridis'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


m0a = VGG(dataset='CIFAR10',depth=19)
m0b = VGG(dataset='CIFAR10',depth=19)
m1a = VGG(dataset='CIFAR10',depth=19)
m1b = VGG(dataset='CIFAR10',depth=19)
m2a = VGG(dataset='CIFAR10',depth=19)
m2b = VGG(dataset='CIFAR10',depth=19)
m3a = VGG(dataset='CIFAR10',depth=19)
m3b = VGG(dataset='CIFAR10',depth=19)



#import pdb; pdb.set_trace()

m0a.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.0_variant1_train-frac0.9_steps1_exp_normal_kaiming_batch1_rseed_481_cross_entropy_20.model'))
m0a.eval()

m1a.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.9_variant1_train-frac0.9_steps1_exp_normal_kaiming_batch1_rseed_37_cross_entropy_20.model'))
m1b.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.99_variant1_train-frac0.9_steps1_exp_normal_kaiming_batch1_rseed_149_cross_entropy_20.model'))
m1a.eval()
m1b.eval()
m2a.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.9_variant1_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_230_cross_entropy_20.model'))
m2b.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.99_variant1_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_212_cross_entropy_20.model'))
m2a.eval()
m2b.eval()
m3a.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.9_variant3_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_250_cross_entropy_20.model'))
m3b.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.99_variant3_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_270_cross_entropy_20.model'))
m3a.eval()
m3b.eval()
#m4.load_state_dict(torch.load('saved_models/vgg19_CIFAR10_spars0.9_variant1_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_810_cross_entropy_50.model'))
# m4.eval()
m0b.load_state_dict(torch.load('useful/vgg19_CIFAR10_spars0.0_variant3_train-frac0.9_steps60_exp_normal_kaiming_batch1_rseed_251_cross_entropy_20.model'))
m0b.eval()

print('worked')

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

inv_normalize = Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1944, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010],
)
normalize = torchvision.transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    normalize,
])
data_dir = '~/data'
dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False,
    download=True, transform=transform,
)
cifar_test = torch.utils.data.DataLoader(dataset, batch_size=1)

y_lbls = np.zeros(len(cifar_test))
y_pred0a = np.zeros(len(cifar_test))
y_pred0b = np.zeros(len(cifar_test))
y_pred1a = np.zeros(len(cifar_test))
y_pred1b = np.zeros(len(cifar_test))
y_pred2a = np.zeros(len(cifar_test))
y_pred2b = np.zeros(len(cifar_test))
y_pred3a = np.zeros(len(cifar_test))
y_pred3b = np.zeros(len(cifar_test))
#testloader = get_cifar_test_loader(1)

#m0.forward()

## Visualize feature maps
#activation = {}
#def get_activation(name):
#    def hook(model, input, output):
#        activation[name] = output.detach()
#    return hook
#
#m0.conv1.register_forward_hook(get_activation('conv1'))
#data, _ = dataset[0]
#data.unsqueeze_(0)
#output = m0(data)
#
#act = activation['conv1'].squeeze()
#fig, axarr = plt.subplots(act.size(0))
#for idx in range(act.size(0)):
#    axarr[idx].imshow(act[idx])
#
#import pdb; pdb.set_trace()

for num,(img,lbl) in enumerate(cifar_test):
    y_lbls[num] = lbl.detach().numpy()
    out0a = m0a.forward(img)
    _, pred0a = torch.max(out0a, 1)
    y_pred0a[num] = pred0a.detach().numpy()
    out0b = m0b.forward(img)
    _, pred0b = torch.max(out0b, 1)
    y_pred0b[num] = pred0b.detach().numpy()
    out1a = m1a.forward(img)
    _, pred1a = torch.max(out1a, 1)
    y_pred1a[num] = pred1a.detach().numpy()
    out1b = m1b.forward(img)
    _, pred1b = torch.max(out1b, 1)
    y_pred1b[num] = pred1b.detach().numpy()
    out2a = m2a.forward(img)
    _, pred2a = torch.max(out2a, 1)
    y_pred2a[num] = pred2a.detach().numpy()
    out2b = m2b.forward(img)
    _, pred2b = torch.max(out2b, 1)
    y_pred2b[num] = pred2b.detach().numpy()
    out3a = m3a.forward(img)
    _, pred3a = torch.max(out3a, 1)
    y_pred3a[num] = pred3a.detach().numpy()
    out3b = m3b.forward(img)
    _, pred3b = torch.max(out3b, 1)
    y_pred3b[num] = pred3b.detach().numpy()

    #fmap1 = m1a.feature[0](img)
    #fmap1 = fmap1.squeeze()

    #square = 8
    #ix = 1
    #for _ in range(square):
	#    for _ in range(square):
	#	    # specify subplot and turn of axis
	#	    ax = plt.subplot(square, square, ix)
	#	    ax.set_xticks([])
	#	    ax.set_yticks([])
	#	    # plot filter channel in grayscale
	#	    plt.imshow(torchvision.transforms.ToPILImage()(fmap1[ix-1]), interpolation="bicubic")
	#	    ix += 1
    ## show the figure
    #plt.show()
    #print(pred)
    #import pdb; pdb.set_trace()

examples = np.zeros([10,1000])
for i in range(10):
    index = np.where(y_lbls == i)
    tmp = np.asarray(index)
    tmp = tmp.squeeze()
    examples[i,:] = tmp

cm_plot_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cm0a = confusion_matrix(y_true=y_lbls, y_pred=y_pred0a)# labels=cm_plot_labels, normalize=normalize)
cm0b = confusion_matrix(y_true=y_lbls, y_pred=y_pred0b)# labels=cm_plot_labels, normalize=normalize)

cm1a = confusion_matrix(y_true=y_lbls, y_pred=y_pred1a)# labels=cm_plot_labels, normalize=normalize)
cm1b = confusion_matrix(y_true=y_lbls, y_pred=y_pred1b)# labels=cm_plot_labels, normalize=normalize)

cm2a = confusion_matrix(y_true=y_lbls, y_pred=y_pred2a)# labels=cm_plot_labels, normalize=normalize)
cm2b = confusion_matrix(y_true=y_lbls, y_pred=y_pred2b)# labels=cm_plot_labels, normalize=normalize)

cm3a = confusion_matrix(y_true=y_lbls, y_pred=y_pred3a)# labels=cm_plot_labels, normalize=normalize)
cm3b = confusion_matrix(y_true=y_lbls, y_pred=y_pred3b)# labels=cm_plot_labels, normalize=normalize)

s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b = 0, 0, 0, 0, 0, 0, 0, 0
for i in range(10):
    s0a += cm0a[i,i]/10000
    s0b += cm0b[i,i]/10000
    
    s1a += cm1a[i,i]/10000
    s1b += cm1b[i,i]/10000

    s2a += cm2a[i,i]/10000
    s2b += cm2b[i,i]/10000

    s3a += cm3a[i,i]/10000
    s3b += cm3b[i,i]/10000

print("s0: {}, s1a: {}, s1b: {}, s2a: {}, s2b: {}, s3a: {}, s3b: {}".format(s0a,s0b,s1a,s1b,s2a,s2b,s3a,s3b))


plot1 = plt.figure(1)
plot_confusion_matrix(cm=cm0a, classes=cm_plot_labels, title='Confusion Matrix 0', normalize=True)
plot2 = plt.figure(2)
plot_confusion_matrix(cm=cm0b, classes=cm_plot_labels, title='Confusion Matrix 0', normalize=True)
plot3 = plt.figure(3)
plot_confusion_matrix(cm=cm1a, classes=cm_plot_labels, title='Confusion Matrix 1a', normalize=True)
plot4 = plt.figure(4)
plot_confusion_matrix(cm=cm1b, classes=cm_plot_labels, title='Confusion Matrix 1b', normalize=True)
plot5 = plt.figure(5)
plot_confusion_matrix(cm=cm2a, classes=cm_plot_labels, title='Confusion Matrix 2a', normalize=True)
plot6 = plt.figure(6)
plot_confusion_matrix(cm=cm2b, classes=cm_plot_labels, title='Confusion Matrix 2b', normalize=True)
plot7 = plt.figure(7)
plot_confusion_matrix(cm=cm3a, classes=cm_plot_labels, title='Confusion Matrix 3a', normalize=True)
plot8 = plt.figure(8)
plot_confusion_matrix(cm=cm3b, classes=cm_plot_labels, title='Confusion Matrix 3b', normalize=True)
plt.show()

import pdb; pdb.set_trace()