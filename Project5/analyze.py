# 24 Fall
# CS 5330 - Project 5
# Sihe Chen
# 002085773
# chen.sihe1@northeastern.edu
# analyze the trained network

import argparse
import torch
from mnist import get_model, load_test_dataset, kDevice
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


# visualize the parameters of the first layer of the network
def show_first_layer(network):
    network.eval()
    with torch.no_grad():
        first_layer = network.conv1.weight.cpu()

    fig, axes = plt.subplots(2, 5)

    for i in range(10):
        image = first_layer[i].squeeze()
        axes[i // 5][i % 5].imshow(image, cmap="gray")
        axes[i // 5][i % 5].set_title(f"layer: {i + 1}", fontsize=8)
        axes[i // 5][i % 5].axis("off")

    plt.tight_layout()
    plt.savefig("results/first_layer.jpg")


# apply the first layer filters to the image
def apply_filter_to_image(network, test_loader):
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(kDevice), target.to(kDevice)
            # data.shape: 1000, 1, 28, 28 (batch size is 1000)
            image = data[0]
            filtered = network.conv1(image).cpu()
            # filtered.shape: 10, 24, 24

            first_layer = network.conv1.weight.cpu()
            fig, axes = plt.subplots(5, 4)
            for i in range(10):
                image = filtered[i].squeeze()
                weight = first_layer[i].squeeze()
                axes[i // 2][i % 2 * 2].imshow(image, cmap="gray")
                axes[i // 2][i % 2 * 2].axis("off")
                axes[i // 2][i % 2 * 2 + 1].imshow(weight, cmap="gray")
                axes[i // 2][i % 2 * 2 + 1].axis("off")

            # Display the plot
            plt.tight_layout()
            plt.savefig("results/apply_filter.jpg")

            break
    pass


# main function
def main(args):
    network = get_model()
    show_first_layer(network)
    test_loader = load_test_dataset()
    apply_filter_to_image(network, test_loader)


if __name__ == "__main__":
    main(parse_args())
