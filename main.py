import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import KeyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'best_model.pth'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'audio'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
n_epochs = 100
batch_size = 1
num_label = 24
method = 'logmelspectrogram'
sr = 22050
momentum = 0.9
weight_decay = 1e-4


class KeyNet(nn.Module):
    def __init__(self):
        super(KeyNet, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1)
        self.batch_norm4 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1)
        self.batch_norm5 = nn.BatchNorm2d(40)

        self.conv5 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1)
        self.batch_norm6 = nn.BatchNorm2d(80)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1)
        self.batch_norm7 = nn.BatchNorm2d(80)
        self.conv7 = nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3, stride=1)
        self.batch_norm8 = nn.BatchNorm2d(160)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1)
        self.batch_norm9 = nn.BatchNorm2d(160)
        self.conv9 = nn.Conv2d(in_channels=160, out_channels=24, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # print("연산 전", x.size())
        x = self.batch_norm1(x)
        x = F.elu(self.conv1(x))
        x = self.batch_norm2(x)
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.batch_norm3(x)
        x = F.elu(self.conv3(x))
        x = self.batch_norm4(x)
        x = F.elu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.batch_norm5(x)
        x = F.elu(self.conv5(x))
        x = self.batch_norm6(x)
        x = F.elu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.batch_norm7(x)
        x = F.elu(self.conv7(x))
        x = self.dropout(x)

        x = self.batch_norm8(x)
        x = F.elu(self.conv8(x))
        x = self.dropout(x)

        x = self.batch_norm9(x)
        x = F.elu(self.conv9(x))
        x = self.avgpool(x)
        x = x.squeeze()

        return x
# # TODO : Build your model here
# class View(nn.Module):

#     def __init__(self, *shape):
#         super(View, self).__init__()
#         self.shape = shape

#     def forward(self, x):
#         return x.view(x.shape[0], *self.shape)

# class Residual_Block(nn.Module):

#     def __init__(self, n_ch):
#         super(Residual_Block, self).__init__()
#         layers = []
#         layers += [nn.BatchNorm2d(num_features=n_ch),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=3, stride=1, padding=1, bias=False),
#                   nn.BatchNorm2d(num_features=n_ch),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=3, stride=1, padding=1, bias=False)]
#         self.layers = nn.Sequential(*layers)

#     def forward(self,x):
#         out = self.layers(x)
#         return x + out

# class ResNet(nn.Module):

#     def __init__(self):
#         super(ResNet, self).__init__()

#         layers = []
#         layers += [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#                    Residual_Block(n_ch=64),
#                    Residual_Block(n_ch=64),
#                    nn.BatchNorm2d(64),
#                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
#                    Residual_Block(n_ch=256),
#                    Residual_Block(n_ch=256),
#                    nn.AdaptiveAvgPool2d((1,1)),
#                    View(-1),
#                    nn.Linear(in_features=256, out_features=24)]

#         self.layers = nn.Sequential(*layers)

#     def forward(self,x):
#         return self.layers(x)


if not is_test_mode:

    # Load Dataset and Dataloader
    train_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='training')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='validation')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Define Model, loss, optimizer
    model = KeyNet()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9, weight_decay=1e-4)


    # Training and Validation
    for epoch in range(n_epochs):

        model.train()

        train_correct = 0
        train_loss = 0

        for idx, (features, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            features = signal_process(features, sr=sr, method=method).to(device)
            features = features.unsqueeze(1)
            labels = labels.to(device)

            output = model(features)
            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=-1, keepdim=True)
            train_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Train Loss: %.2f, Train Accuracy: %.3f" % (
            epoch, train_loss / len(train_loader), train_correct / len(train_dataset)))

        model.eval()

        valid_correct = 0
        valid_loss = 0

        for idx, (features, labels) in enumerate(valid_loader):
            features = signal_process(features, sr=sr, method=method).to(device)
            features = features.unsqueeze(1)
            labels = labels.to(device)

            output = model(features)
            loss = criterion(output, labels)
            valid_loss += loss.item()

            preds = output.argmax(dim=-1, keepdim=True)
            valid_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Valid Loss: %.2f, Valid Accuracy: %.3f" % (
            epoch, valid_loss / len(valid_loader), valid_correct / len(valid_dataset)))

elif is_test_mode:

    # Load Dataset and Dataloader
    test_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='validation')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Restore model
    model = KeyNet()
    model.load_state_dict(torch.load(restore_path))
    model = model.to(device)
    print('==== Model restored : %s' % restore_path)

    # TODO: IMPORTANT!! MUST CALCULATE ACCURACY ! You may change this part, but must print accuracy in the right manner
    test_correct = 0

    for features, labels in test_loader:
        features = signal_process(features, sr=sr, method=method).to(device)
        print(features.shape)
        features = features.unsqueeze(1)
        print(features.shape)
        labels = labels.to(device)
        output = model(features)
        preds = output.argmax(dim=-1, keepdim=True)
        test_correct += (preds.squeeze() == labels).float().sum()

    print("=== Test accuracy: %.3f" % (test_correct / len(test_dataset)))
