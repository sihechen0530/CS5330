# 24 Fall
# CS 5330 - Project 5
# Sihe Chen
# 002085773
# chen.sihe1@northeastern.edu
# evaluate the deep network trained with mnist.py

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms

from mnist import (
    get_model,
    load_test_dataset,
    load_network,
    kDevice,
)


# parse arguments from command line
def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


# evaluate on test dataset
def evaluate_with_test_data(network):
    network.eval()
    kExampleCount = 10
    test_loader = load_test_dataset()
    images, labels = next(iter(test_loader))[:kExampleCount]

    with torch.no_grad():
        output = network(images.to(kDevice))

    fig = plt.figure()
    for i in range(kExampleCount):
        plt.subplot(5, 2, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap="gray", interpolation="none")
        plt.title(
            "Prediction: {} GT: {}".format(
                output.data.max(1, keepdim=True)[1][i].item(), labels[i]
            ),
        )
        plt.xticks([])
        plt.yticks([])
    fig.savefig("results/evaluate_with_test_data.jpg")


# evaluate with new handwritten image
def evaluate_with_image(network):
    network.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize image to 28x28 pixels
            transforms.ToTensor(),  # Convert to tensor and scale pixel values to [0, 1]
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize with mean and std (adjust if needed)
        ]
    )
    kFileDir = "hand_written"
    kNums = 10
    images = []
    for i in range(kNums):
        file_path = f"{kFileDir}/{i}.png"
        image = Image.open(file_path).convert("L")
        image = ImageOps.invert(image)
        image = transform(image)
        images.append(image)

    with torch.no_grad():
        output = network(torch.stack(images).to(kDevice))

    fig = plt.figure()
    for i in range(kNums):
        plt.subplot(5, 2, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap="gray", interpolation="none")
        plt.title(
            "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()),
        )
        plt.xticks([])
        plt.yticks([])
    fig.savefig("results/evaluate_with_hand_written.jpg")


# main function
def main(args):
    network, optimizer = load_network()
    evaluate_with_test_data(network)
    evaluate_with_image(network)


if __name__ == "__main__":
    main(parse_args())

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig


# continued training from checkpoints
# continued_network = Net()
# continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                                 momentum=momentum)


# network_state_dict = torch.load()
# continued_network.load_state_dict(network_state_dict)

# optimizer_state_dict = torch.load()
# continued_optimizer.load_state_dict(optimizer_state_dict)

# for i in range(4,9):
#   test_counter.append(i*len(train_loader.dataset))
#   train(i)
#   test()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig
