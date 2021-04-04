import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, channels_in, out_dim):
        super(DQN, self).__init__()
        # visual features
        # channels x dim_x, dim_y
        self.conv1 = nn.Conv2d(channels_in, 3, 3)
        self.bn1 = nn.BatchNorm2d(3)
        # 2x2 kernel ==
        # self.conv2 = nn.Conv2d(3, 3, 3)
        # self.bn2 = nn.BatchNorm2d(3)
        # self.conv3 = nn.Conv2d(12, 4, 3)
        # self.bn3 = nn.BatchNorm2d(4)

        # dropout, important during small size of gathered Transitions containing a reward
        self.dropout = nn.Dropout(p=0.01, inplace=False)

        # predict Q-value for specific action
        # (in_channel, dim_x, dim_y) where dim_x and dim_y are downsampled
        self.fc1 = nn.Linear(3 * 31 * 31, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, out_dim)

    def forward(self, screens):
        # https://discuss.pytorch.org/t/runtimeerror-expected-scalar-type-int-but-found-float/102140
        screens = screens.float()
        conv_output = F.max_pool2d(F.relu(self.bn1(self.conv1(screens))), (2, 2))
        # conv_output = F.max_pool2d(F.relu(self.bn2(self.conv2(conv_output))), (2, 2))
        # conv_output = F.relu(self.bn3(self.conv3(conv_output))), (2, 2)
        conv_output = conv_output.view(-1, self.num_flat_features(conv_output))
        x = F.relu(self.fc1(conv_output))
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        pred_action = F.relu(self.fc3(x))
        return pred_action

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
