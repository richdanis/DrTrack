import torch
import torchvision
import os


class EfficientNet(torch.nn.Module):

    def __init__(self, cfg):
        model_args = cfg.extract_visual_embeddings
        checkpoint_path = os.path.join(cfg.checkpoint_dir, model_args.checkpoint)
        super().__init__()

        # initialize with pretrained weights
        # does not work on euler!
        # download model yourself and transfer to euler
        # self.layers = torchvision.models.efficientnet_b1(weights='DEFAULT')

        # load from local state dict
        self.layers = torchvision.models.efficientnet_b1()

        # change classifier to identity
        self.layers.classifier = torch.nn.Identity()

        if model_args.embed_dim is not None:
            # add embedding layer
            self.layers.classifier = torch.nn.Linear(1280, model_args.embed_dim)

        self.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device(cfg.device))['model_state_dict'])

    def forward(self, x):
        assert len(x.shape) == 4, "Inputs must be images of shape (N, 2, H, W)"
        # repeat along first dimension
        # this tutorial explains nicely why this is necessary:
        # https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a
        x = torch.cat((x, x[:, :1, :, :]), dim=1)

        return self.layers(x)
