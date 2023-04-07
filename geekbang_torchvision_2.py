import torchvision
def main():
    mnist_dataset =torchvision.datasets.MNIST(root="./data",
                                         train=True,
                                         transform=None,
                                         target_transform=None,
                                         download=True)
if __name__ == '__main__':
        main()

