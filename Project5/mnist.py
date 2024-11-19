# 24 Fall
# CS 5330 - Project 5
# Sihe Chen
# 002085773
# chen.sihe1@northeastern.edu
# train a deep network on mnist dataset

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


kModelPath = "results/model.pth"
kOptimizerPath = "results/optimizer.pth"
kLearningRate = 0.01
kMomentum = 0.5
kBatchTestSize = 1000
kLogInterval = 10
kDevice = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_train_size", type=int)
    return parser.parse_args()


# Self-defined Neural Network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # dropout default probability is 0.5 (p=0.5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# initialize a model
def get_model():
    return MyNetwork().to(kDevice)


# load model parameters from an existing file
def load_network():
    network = get_model()
    optimizer = optim.SGD(network.parameters(), lr=kLearningRate, momentum=kMomentum)

    network_state_dict = torch.load(kModelPath)
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load(kOptimizerPath)
    optimizer.load_state_dict(optimizer_state_dict)
    return network, optimizer


# load training dataset
def load_train_dataset(batch_size_train):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )
    return train_loader


# load testing dataset
def load_test_dataset():
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=kBatchTestSize,
        shuffle=True,
    )
    return test_loader


# plot test examples and save as an image file
def show_test_examples(test_loader):
    images, labels = next(iter(test_loader))

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(6):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Label: {labels[i]}")

    plt.tight_layout()
    # plt.show()
    plt.savefig("results/test_examples.jpg")


# compute accuracy of predicted result
def calculate_accuracy(output, labels):
    _, predicted = torch.max(output, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


# train a network and save losses and accuracies
def train(
    epoch,
    network,
    train_loader,
    optimizer,
    accuracy,
    train_losses,
    train_counter,
    log_interval=kLogInterval,
    model_path=kModelPath,
    optimizer_path=kOptimizerPath,
):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(kDevice), target.to(kDevice)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        acc = calculate_accuracy(output, target)
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    acc,
                )
            )
            accuracy.append(acc)
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(network.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)


# test a network on testing dataset and save testing losses
def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(kDevice), target.to(kDevice)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


# plot the losses and accuracies and save as an image file
def plot(
    network,
    epochs,
    train_loader,
    train_losses,
    train_counter,
    accuracy,
    test_losses=None,
    image_path="losses.jpg",
):
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]
    plt.subplot(1, 2, 1)
    plt.plot(train_counter, train_losses, color="green")
    if test_losses:
        plt.scatter(test_counter, test_losses, color="orange")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_counter, accuracy, label="Training Accuracy", color="blue")
    plt.xlabel("number of training examples seen")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(image_path)


# train the network with batch size and number of epochs
def train_network(batch_train_size, n_epochs):
    train_loader = load_train_dataset(batch_train_size)
    network = get_model()
    optimizer = optim.SGD(network.parameters(), lr=kLearningRate, momentum=kMomentum)

    accuracy = []
    train_losses = []
    train_counter = []

    test_loader = load_test_dataset()

    # show_test_examples(test_loader)
    # exit(0)

    test_losses = []
    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(
            epoch,
            network,
            train_loader,
            optimizer,
            accuracy,
            train_losses,
            train_counter,
        )
        test(network, test_loader, test_losses)

    plot(
        network,
        n_epochs,
        train_loader,
        train_losses,
        train_counter,
        accuracy,
        test_losses,
    )


# main function
def main(args):
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_network(args.batch_train_size, args.epoch)
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
