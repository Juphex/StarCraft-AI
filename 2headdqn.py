import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, channels_in, out_dim):
        super(DQN, self).__init__()
        # visual features
        self.conv1 = nn.Conv2d(channels_in, 18, 3)
        self.bn1 = nn.BatchNorm2d(18)
        self.conv2 = nn.Conv2d(18, 6, 3)
        self.bn2 = nn.BatchNorm2d(6)

        # predict Q-value for specific action
        # 6 is the out channels, 15 * 15 is the downsampled image dimension
        self.fc1 = nn.Linear(6 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, screens):
        # https://discuss.pytorch.org/t/runtimeerror-expected-scalar-type-int-but-found-float/102140
        screens = screens.float()
        conv_output = F.max_pool2d(F.relu(self.bn1(self.conv1(screens))), (2, 2))
        conv_output = F.max_pool2d(F.relu(self.bn2(self.conv2(conv_output))), (2, 2))
        conv_output = conv_output.view(-1, self.num_flat_features(conv_output))
        x = F.relu(self.fc1(conv_output))
        x = F.relu(self.fc2(x))
        pred_action = F.relu(self.fc3(x))
        return pred_action

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features