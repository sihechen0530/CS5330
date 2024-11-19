# 24 Fall
# CS 5330 - Project 5
# Sihe Chen
# 002085773
# chen.sihe1@northeastern.edu
# run MNIST experiment with different combinations of hyper parameters

import argparse
import torch
import json
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
kDevice = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {kDevice} device")
kBatchTestSize = 1000
str2list = lambda x: list(map(int, str(x).split(",")))
kImageShape = (28, 28)


# get activation function by name
def get_activation_function(function_name="relu"):
    func_map = {"tanh": torch.tanh, "relu": F.relu, "sigmoid": nn.Sigmoid()}
    return func_map[function_name]


# get loss function by name
def get_loss_function(function_name="nll"):
    func_map = {"nll": F.nll_loss, "cross_entropy": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}
    return func_map[function_name]


# parse commandline argument
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_train_size", type=int, default=64)
    parser.add_argument("--num_conv_layers", type=int, default=2)
    parser.add_argument("--num_conv_filters", type=str, default="10,20")
    parser.add_argument("--conv_filter_sizes", type=str, default="5,5")
    parser.add_argument("--num_hidden_nodes", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--loss_function", type=str, default="nll")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--exp_id", type=int)
    return parser.parse_args()


# load training data
def load_train_data(batch_size):
    train_dataloader = DataLoader(
        torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
    )
    return train_dataloader


# load testing data
def load_test_data(batch_size):
    test_dataloader = DataLoader(
        torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
    )
    return test_dataloader


# Define model
class CustomizedNetwork(nn.Module):
    def __init__(
        self,
        num_conv_layers=2,
        num_conv_filters=[10, 20],
        conv_filter_sizes=[5, 5],
        num_hidden_nodes=50,
        dropout_rate=0.5,
        activation_function=F.relu,
    ):
        super(CustomizedNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        layer_shape = kImageShape
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels, num_conv_filters[i], kernel_size=conv_filter_sizes[i]
                )
            )
            in_channels = num_conv_filters[i]
            layer_shape = (
                (layer_shape[0] - conv_filter_sizes[i] // 2 * 2) // 2,
                (layer_shape[1] - conv_filter_sizes[i] // 2 * 2) // 2,
            )
        fc_input_size = num_conv_filters[-1] * layer_shape[0] * layer_shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, 10)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_function = activation_function

    def forward(self, x):
        # Pass through convolutional layers with activation and pooling
        for conv in self.conv_layers:
            x = self.activation_function(F.max_pool2d(conv(x), 2))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with activation and dropout
        x = self.activation_function(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x)


# train network and evaluate
def train_and_evaluate(
    network,
    train_loader,
    test_loader,
    num_epochs,
    criterion,
    optimizer,
    model_path="default_model.pth",
    optimizer_path="default_optimizer.pth",
):
    train_losses, eval_losses = [], []
    train_accuracies, eval_accuracies = [], []

    for epoch in range(num_epochs):
        # Training Phase
        network.train()
        total_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(kDevice), labels.to(kDevice)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train_preds += (preds == labels).sum().item()
            total_train_samples += labels.size(0)

        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(correct_train_preds / total_train_samples)

        torch.save(network.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

        # Validation Phase
        network.eval()
        total_val_loss = 0
        correct_val_preds = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(kDevice), labels.to(kDevice)
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        eval_losses.append(total_val_loss / len(test_loader))
        eval_accuracies.append(correct_val_preds / total_val_samples)

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
            f"Val Loss: {eval_losses[-1]:.4f}, Val Acc: {eval_accuracies[-1]:.4f}"
        )

    return train_losses, eval_losses, train_accuracies, eval_accuracies


# plot losses and accuracies
def plot_metrics(
    train_losses,
    eval_losses,
    train_accuracies,
    eval_accuracies,
    image_path="default_image.jpg",
):
    # Plot Loss
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(eval_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()

    plt.savefig(image_path)


# main function
def main(args):
    # load data
    train_loader = load_train_data(args.batch_train_size)
    test_loader = load_test_data(kBatchTestSize)
    # init model with args
    network = CustomizedNetwork(
        args.num_conv_layers,
        str2list(args.num_conv_filters),
        str2list(args.conv_filter_sizes),
        args.num_hidden_nodes,
        args.dropout_rate,
        get_activation_function(args.activation_function),
    ).to(kDevice)
    print(network)
    optimizer = torch.optim.SGD(
        network.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    # train and evaluate
    identification = f"experiments/{args.exp_id}_"
    arguments = [
        args.num_conv_layers,
        args.num_conv_filters,
        args.conv_filter_sizes,
        args.num_hidden_nodes,
        args.dropout_rate,
        args.activation_function,
        args.loss_function,
    ]
    for arg in arguments:
        identification += f"{arg}_"
    model_path = f"{identification}model.pth"
    optimizer_path = f"{identification}optimizer.pth"
    train_losses, eval_losses, train_accuracies, eval_accuracies = train_and_evaluate(
        network,
        train_loader,
        test_loader,
        args.num_epochs,
        get_loss_function(args.loss_function),
        optimizer,
        model_path,
        optimizer_path,
    )
    # plot fig
    plot_metrics(
        train_losses,
        eval_losses,
        train_accuracies,
        eval_accuracies,
        f"{identification}vis.jpg",
    )
    # save raw data to file
    with open(f"{identification}raw.json", "w") as f:
        f.write(
            json.dumps(
                [arguments, train_losses, eval_losses, train_accuracies, eval_accuracies], indent=4
            )
        )


if __name__ == "__main__":
    main(parse_args())
