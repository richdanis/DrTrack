import torch
import torchvision


class EfficientNet(torch.nn.Module):

    def __init__(self, args, path='src/training/models/efficientnet_b1.pth'):
        super().__init__()

        # initialize with pretrained weights
        # does not work on euler!
        # download model yourself and transfer to euler
        # self.layers = torchvision.models.efficientnet_b1(weights='DEFAULT')
        self.args = args

        # load from local state dict
        self.layers = torchvision.models.efficientnet_b1()
        self.layers.load_state_dict(torch.load(path))

        # change classifier to identity
        self.layers.classifier = torch.nn.Identity()

        if args.embed_dim is not None:
            # add embedding layer
            self.layers.classifier = torch.nn.Linear(1280, args.embed_dim)

    def forward(self, x):
        assert len(x.shape) == 4, "Inputs must be grayscale images of shape (N, 1, H, W)"
        # repeat along first dimension
        # this tutorial explains nicely why this is necessary:
        # https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a
        if self.args.use_dapi:
            # concatenate first channel to make it three channels
            x = torch.cat((x, x[:, :1, :, :]), dim=1)
        else:
            x = x.repeat(1, 3, 1, 1)

        return self.layers(x)


def save_locally():
    # download model
    model = torchvision.models.efficientnet_b1(weights='DEFAULT')
    # save model
    torch.save(model.state_dict(), 'src/training/models/efficientnet_b1.pth')


if __name__ == "__main__":
    save_locally()
