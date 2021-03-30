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

    # class Model(nn.Module):
    #     def __init__(self, channels_in, out_dim):
    #         super(Model, self).__init__()
    #         # visual features
    #         self.conv1 = nn.Conv2d(channels_in, 64, 12)
    #         self.bn1 = nn.BatchNorm2d(64)
    #         self.conv2 = nn.Conv2d(64, 32, 3)
    #         self.bn2 = nn.BatchNorm2d(32)
    #
    #         # predict Q-value for specific action
    #         # 12 * 12 is the downsampled image dimension
    #         self.fc1 = nn.Linear(32 * 12 * 12, 512)
    #         self.fc2 = nn.Linear(512, 256)
    #         self.fc3 = nn.Linear(256, out_dim)
    #
    #         # predict possible args for all actions
    #         self.fc_arg1_1 = nn.Linear((32 * 12 * 12) + out_dim, 256)
    #         self.fc_arg1_2 = nn.Linear(256, 1)
    #
    #         self.fc_arg2_1 = nn.Linear((32 * 12 * 12) + out_dim, 256)
    #         self.fc_arg2_2 = nn.Linear(256, 1)
    #
    #         self.fc_arg3_1 = nn.Linear((32 * 12 * 12) + out_dim, 256)
    #         self.fc_arg3_2 = nn.Linear(256, 1)
    #
    #         self.fc_arg4_1 = nn.Linear((32 * 12 * 12) + out_dim, 256)
    #         self.fc_arg4_2 = nn.Linear(256, 1)
    #
    #         self.fc_arg5_1 = nn.Linear((32 * 12 * 12) + out_dim, 256)
    #         self.fc_arg5_2 = nn.Linear(256, 1)
    #
    #     def forward(self, feature_screen, feature_minimap):
    #         # shift dimensions => (channel, y, x)
    #         feature_screen = torch.tensor(feature_screen).permute(0, 2, 1)
    #         feature_minimap = torch.tensor(feature_minimap).permute(0, 2, 1)
    #         # concat screens to one tensor
    #         screens = torch.cat((feature_screen, feature_minimap), 0)
    #         # create batch with size 1
    #         screens = screens.unsqueeze(0)
    #         # https://discuss.pytorch.org/t/runtimeerror-expected-scalar-type-int-but-found-float/102140
    #         screens = screens.float()
    #
    #         conv_output = F.max_pool2d(F.relu(self.bn1(self.conv1(screens))), (2, 2))
    #         conv_output = F.max_pool2d(F.relu(self.bn2(self.conv2(conv_output))), (2, 2))
    #         conv_output = conv_output.view(-1, self.num_flat_features(conv_output))
    #         x = F.relu(self.fc1(conv_output))
    #         x = F.relu(self.fc2(x))
    #         pred_action = self.fc3(x)
    #
    #         # flatten pred action TODO
    #         # argument 1
    #         arg_input = torch.cat((conv_output, pred_action.view(-1, self.num_flat_features(pred_action))), 1)
    #         x = F.relu(self.fc_arg1_1(arg_input))
    #         args1 = F.softmax(self.fc_arg1_2(x), )
    #         # argument 2
    #         x = F.relu(self.fc_arg2_1(arg_input))
    #         args2 = F.softmax(self.fc_arg2_2(x), )
    #         # argument 3
    #         x = F.relu(self.fc_arg3_1(arg_input))
    #         args3 = F.softmax(self.fc_arg3_2(x), )
    #         # argument 4
    #         x = F.relu(self.fc_arg4_1(arg_input))
    #         args4 = F.softmax(self.fc_arg4_2(x), )
    #         # argument 5
    #         x = F.relu(self.fc_arg5_1(arg_input))
    #         args5 = F.softmax(self.fc_arg5_2(x), )
    #
    #         return pred_action, args1, args2, args3, args4, args5
    #
    #     # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    #     def num_flat_features(self, x):
    #         size = x.size()[1:]  # all dimensions except the batch dimension
    #         num_features = 1
    #         for s in size:
    #             num_features *= s
    #         return num_features
