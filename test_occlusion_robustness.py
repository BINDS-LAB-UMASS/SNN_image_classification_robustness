import torch
from bindsnet.conversion import ann_to_snn
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from bindsnet.network.monitors import Monitor
from time import time as t_
import pandas as pd
import argparse

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    print("Cuda is available")
else:
    device = torch.device('cpu')
    print("Cuda is not available")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def main(occlusion_percentage=0):

    percentile = 99.9
    random_seed = 0
    torch.manual_seed(random_seed)

    time = 100

    class RandomlyOcclude(object):

        def __init__(self, percentage):
            self.percentage = percentage/100

        def __call__(self, img):
            mask = torch.Tensor(img.shape[1], img.shape[2]).uniform_() > self.percentage
            return img.to(device) * mask.float()



    train_dataset2 = datasets.MNIST('./data',
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor(), RandomlyOcclude(occlusion_percentage)]))


    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())

    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True, batch_size=train_dataset.__len__())

    for d, target in train_loader:
        data = d.to(device)


    model = torch.load('trained_model.pt')

    print()
    print('Converting ANN to SNN...')


    SNN = ann_to_snn(model, input_shape=[28*28], data=data, percentile=percentile)


    SNN.add_monitor(
        Monitor(SNN.layers['2'], state_vars=['s', 'v'], time=time), name='2'
    )



    correct = 0


    def validate():
        model.eval()
        val_loss, correct = 0, 0
        for data, target in train_loader2:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()


        ANN_accuracy = 100. * correct.to(torch.float32) / len(train_loader2.dataset)

        print("ANN accuracy:", ANN_accuracy)
        return ANN_accuracy

    ANN_accuracy = validate()


    start = t_()
    for index, (data, target) in enumerate(train_loader2):
        print('sample ', index+1, 'elapsed', t_() - start)
        start = t_()

        data = data.to(device)
        data = data.view(-1, 28*28)
        inpts = {'Input': data.repeat(time, 1)}
        SNN.run(inpts=inpts, time=time)
        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors if not layer == 'Input'}
        pred = torch.argmax(voltages['2'].sum(1))
        correct += pred.eq(target.data.to(device)).cpu().sum()
        SNN.reset_()

    SNN_accuracy = 100. * correct.to(torch.float32) / len(train_loader2.dataset)


    print("accuracy:, ", SNN_accuracy)

    df = pd.DataFrame({"ANN accuracy":[ANN_accuracy],
                       "SNN accuracy": [SNN_accuracy]})

    df.to_csv("accuracy_1hidden_"+str(occlusion_percentage)+".csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--occlusion_percentage', type=int, default=0)
    args = vars(parser.parse_args())

    main(**args)