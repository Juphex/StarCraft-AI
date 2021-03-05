import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Softmax()
        )

    def forward(self, feature_screen, feature_minimap):
        # shift dimensions (so last dim is channel) => (x,y,channel)
        feature_screen = torch.tensor(feature_screen).permute(2, 1, 0)
        feature_minimap = torch.tensor(feature_minimap).permute(2, 1, 0)
        screens = torch.cat((feature_screen, feature_minimap), 2)
        print(screens.size())
        #CONV net
        pred_action = 1
        #  x = self.flatten(x)
        # pred_action = self.linear_relu_stack(x)
        return pred_action
