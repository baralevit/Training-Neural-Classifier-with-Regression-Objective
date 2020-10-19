import torch
import torchvision
import torchvision.transforms as transforms

batch_size =100


transform_train = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

transform_test = transforms.Compose([
    transforms.Pad(4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


classes = ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG19'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 9)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #print(out.size())        
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels =1 
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)





net = VGG()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

print(device)
import torch.optim as optim

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def labels_to_R9(labels):
    matlabels=torch.zeros(batch_size,9)
      
        
    for j in range(batch_size):
        y=labels[j]
        if y==0:
            continue
        else:
            matlabels[j,y-1]=1
    #print (matlabels.size())        
    return matlabels.to(device)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
 
        # zero the parameter gradients
        optimizer.zero_grad()

        
        matlabels=labels_to_R9(labels)
              
      
      
      
        # forward + backward + optimize
        outputs = net(inputs)
                
        loss = criterion(outputs, matlabels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#outputs = net(images)

#_, predicted = torch.max(outputs, 1)

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(batch_size)))

def prediction(outputs):
    alllabels=torch.zeros(10,9)
    for i in range(1,10):
        alllabels[i,i-1]=1
        
    distance=torch.zeros(batch_size,10)
    for i in range(batch_size):
        for j in range(10):
            distance[i,j]=torch.dist(outputs[i],alllabels[j].to(device),2)
     
    _, predicted=torch.min(distance,1)
    return predicted

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        matlabels=labels_to_R9(labels)
        outputs = net(images)

        
        alllabels=torch.zeros(10,9)
        for i in range(1,10):
            alllabels[i,i-1]=1
        
        distance=torch.zeros(batch_size,10)
        for i in range(batch_size):
            for j in range(10):
                distance[i,j]=torch.dist(outputs[i],alllabels[j].to(device),2)
     
        _, predicted=torch.min(distance,1)
     
        #predicted=prediction(outputs)
        
        
        total += matlabels.size(0)
        correct += (predicted.to(device) == labels.to(device)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct =list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        alllabels=torch.zeros(10,9)
        for i in range(1,10):
            alllabels[i,i-1]=1
        
        distance=torch.zeros(batch_size,10)
        for i in range(batch_size):
            for j in range(10):
                distance[i,j]=torch.dist(outputs[i],alllabels[j].to(device),2)
     
        _, predicted=torch.min(distance,1)
      #predicted=prediction(outputs)
        
        c = (predicted.to(device) == labels.to(device)).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

class_outputs = torch.zeros(10,10)
class_total = torch.zeros(10)
confusion_matrix=torch.zeros(10,10)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        
        alllabels=torch.zeros(10,9)
        for i in range(1,10):
            alllabels[i,i-1]=1
        
        distance=torch.zeros(batch_size,10)
        for i in range(batch_size):
            for j in range(10):
                distance[i,j]=torch.dist(outputs[i],alllabels[j].to(device),2)
     
        _, predicted=torch.min(distance,1)
        #predicted=prediction(outputs)
      

        
        for i in range(batch_size):
            class_total[labels[i]] += 1
            class_outputs[labels[i],predicted[i]]+=1
                #print(confusion_matrix[i,j])

#for i in range(10):
#    for j in range(10):
#         confusion_matrix[i,j]=class_outputs[i,j]
        
confusion_matrix=class_outputs.numpy()

print(confusion_matrix)

correct = torch.zeros(10)
total = torch.zeros(10)
precision=torch.zeros(10)
recall=torch.zeros(10)
labeltotal=torch.zeros(10)
outputtotal=torch.zeros(10)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        matlabels=labels_to_R9(labels)
        outputs = net(images)

        
        alllabels=torch.zeros(10,9)
        for i in range(1,10):
            alllabels[i,i-1]=1
        
        distance=torch.zeros(batch_size,10)
        for i in range(batch_size):
            for j in range(10):
                distance[i,j]=torch.dist(outputs[i],alllabels[j].to(device),2)
     
        _, predicted=torch.min(distance,1)
     
        #predicted=prediction(outputs)
        #That is, precision is the fraction of events where we correctly declared
        #$i$ out of all instances where the algorithm declared $i$. Conversely, 
        #recall is the fraction of events where we correctly declared $i$ out of
        #all of the cases where the true of state of the world is $i$
        
        
        #total += matlabels.size(0)
        #correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            #correct += (predicted == labels).sum().item()
            for j in range(10):
                if labels[i]==j:
                    labeltotal[j] +=1
                    correct[j] += (predicted[i].to(device) == labels[i].to(device)).item()
                if predicted[i]==j:
                    outputtotal[j] +=1 
precision=100*correct/outputtotal
precision=precision.numpy()
recall=100*correct/labeltotal
recall=recall.numpy()

print(recall)
print(precision)

