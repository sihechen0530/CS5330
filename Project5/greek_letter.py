import torchvision
import torch
import torch.nn as nn
from mnist import load_network, train, plot, kLearningRate, kMomentum, kDevice
import torch.optim as optim
import argparse
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


kTrainingSetPath = "data/greek_train"
kGreekModelPath = "results/greek.pth"
kGreekOptimizerPath = "results/greek_optimizer.pth"


# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int)
    return parser.parse_args()


# replace the last layer of the network
def replace_last_layer(network):
    # freezes the parameters for the whole network
    print(network)
    for param in network.parameters():
        param.requires_grad = False
    network.fc2 = nn.Linear(50, 3).to(kDevice)
    print(network)


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# load dataset
def load_greek_dataset():
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            kTrainingSetPath,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    GreekTransform(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=5,
        shuffle=True,
    )
    return greek_train


# train the network with n_epochs
def train_network(n_epochs):
    train_loader = load_greek_dataset()
    network, _ = load_network()
    replace_last_layer(network)
    optimizer = optim.SGD(network.parameters(), lr=kLearningRate, momentum=kMomentum)
    accuracy = []
    train_losses = []
    train_counter = []

    # show_test_examples(test_loader)
    # exit(0)

    for epoch in range(1, n_epochs + 1):
        train(
            epoch,
            network,
            train_loader,
            optimizer,
            accuracy,
            train_losses,
            train_counter,
            log_interval=1,
            model_path=kGreekModelPath,
            optimizer_path=kGreekOptimizerPath,
        )
    plot(
        network,
        n_epochs,
        train_loader,
        train_losses,
        train_counter,
        accuracy,
        [],
        "results/greek_losses.jpg",
    )
    return network


# test the network on test data
def test_network(network):
    network.eval()
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    kFileDir = "hand_written"
    images = []
    all_chars = ["alpha", "beta", "gamma"]
    for greek_char in all_chars:
        file_path = f"{kFileDir}/{greek_char}.png"
        image = Image.open(file_path).convert("RGB")
        image = transform(image)
        images.append(image)

    with torch.no_grad():
        output = network(torch.stack(images).to(kDevice))

    fig = plt.figure()
    for i, greek_char in enumerate(all_chars):
        plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap="gray", interpolation="none")
        plt.title(
            "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()),
        )
        plt.xticks([])
        plt.yticks([])
    fig.savefig("results/evaluate_greek_with_hand_written.jpg")


# main function
def main(args):
    network = train_network(args.epoch)
    test_network(network)


if __name__ == "__main__":
    main(parse_args())
