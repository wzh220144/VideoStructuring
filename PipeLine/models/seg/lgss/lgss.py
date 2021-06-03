import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class StftNet(nn.Module):
    def __init__(self, args):
        super(StftNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv2 = nn.Conv2d(64, 192, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,2), padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):  # [bs,1,257,90]
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = x.squeeze()
        out = self.fc(x)
        return out

class Cos(nn.Module):
    def __init__(self, args):
        super(Cos, self).__init__()
        self.window_size = args.window_size
        self.sim_dim = args.sim_dim
        self.conv1 = nn.Conv2d(1, self.sim_dim, kernel_size=(self.window_size, 1))

    def forward(self, x):  # [batch_size, window_size * 2, feat_dim]
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        part1, part2 = torch.split(x, [self.window_size] * 2, dim=2)
        #batch_size, sim_dim, window_size, feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size, sim_dim
        return x

class BNet(nn.Module):
    def __init__(self, args):
        super(BNet, self).__init__()
        self.window_size = args.window_size
        self.sim_dim = args.sim_dim
        self.conv1 = nn.Conv2d(1, self.sim_dim, kernel_size=(self.window_size * 2, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.sim_dim, 1, 1))
        self.cos = Cos(args)

    def forward(self, x):  # [batch_size, window_size * 2, feat_dim]
        context = x.view(-1, 1, x.shape[1], x.shape[2])
        context = self.conv1(context)  # batch_size, sim_dim, 1, feat_dim
        context = self.max3d(context)  # batch_size, 1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound

class BNetSTFT(nn.Module):
    def __init__(self, args):
        super(BNetSTFT, self).__init__()
        self.window_size = args.window_size
        self.sim_dim = args.sim_dim
        self.stft_net = StftNet(args)
        self.conv1 = nn.Conv2d(1, self.sim_dim, kernel_size=(args.window_size * 2, 1))
        self.conv2 = nn.Conv2d(1, self.sim_dim, kernel_size=(args.window_size, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.sim_dim, 1, 1))

    def forward(self, x):  # [batch_size, window_size * 2, 257, 90]
        context = x.view(x.shape[0] * x.shape[1], 1, x.shape[-2], x.shape[-1])
        context = self.stft_net(context)
        context = context.view(x.shape[0], 1, self.window_size * 2, -1)
        part1, part2 = torch.split(context, [self.window_size] * 2, dim=2)
        part1 = self.conv2(part1).squeeze()
        part2 = self.conv2(part2).squeeze()
        sim = F.cosine_similarity(part1, part2, dim=2)
        bound = sim
        return bound

class LGSSone(nn.Module):
    def __init__(self, args, modal):
        super(LGSSone, self).__init__()
        self.window_size = args.window_size
        self.num_layers = args.num_layers
        self.lstm_hidden_size = args.lstm_hidden_size
        if modal == "youtube8m":
            self.bnet = BNet(args)
            self.input_dim = (args.youtube8m_dim + args.sim_dim)
        elif modal == "stft":
            self.bnet = BNetSTFT(args)
            self.input_dim = args.stft_dim
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=args.bidirectional)
        '''
        if args.bidirectional:
            self.fc1 = nn.Linear(self.lstm_hidden_size , 100)
        else:
            self.fc1 = nn.Linear(self.lstm_hidden_size, 100)
        '''
        self.fc1 = nn.Linear(self.input_dim, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.bnet(x)
        x = x.view(-1, x.shape[-1])
        # [batch_size, window_size * 2, 3 * sim_dim]
        #out, (_, _) = self.lstm(x, None)
        #print('out: {}'.format(out.shape))
        # out: tensor of shape (batch_size, window_size * 2, hidden_size)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = out.view(-1, 2)
        return out

class LGSS(nn.Module):
    def __init__(self, args):
        super(LGSS, self).__init__()
        self.extract_youtube8m = args.extract_youtube8m
        self.extract_stft = args.extract_stft
        self.youtube8m_ratio = args.youtube8m_ratio
        self.stft_ratio = args.stft_ratio
        if args.extract_youtube8m:
            self.bnet_youtube8m = LGSSone(args, "youtube8m")
        if args.extract_stft:
            self.bnet_stft = LGSSone(args, "stft")

    def forward(self, youtube8m_feat, stft_feat):
        out = 0
        if self.extract_youtube8m:
            youtube8m_bound = self.bnet_youtube8m(youtube8m_feat)
            out += self.youtube8m_ratio * youtube8m_bound
        if self.extract_stft:
            stft_bound = self.bnet_stft(stft_feat)
            out += self.stft_ratio * stft_bound
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type = str, default = '/home/tione/notebook/VideoStructuring/dataset/samples/seg')
    parser.add_argument('--youtube8m_cache_size', type = int, default = 1000000)
    parser.add_argument('--stft_cache_size', type = int, default = 1000000)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--extract_youtube8m', type = bool, default = True)
    parser.add_argument('--extract_stft', type = bool, default = True)
    parser.add_argument('--youtube8m_ratio', type = float, default = 0.8)
    parser.add_argument('--stft_ratio', type = float, default = 0.2)
    parser.add_argument('--sim_dim', type = int, default = 512)
    parser.add_argument('--youtube8m_dim', type = int, default = 2048)
    parser.add_argument('--stft_dim', type = int, default = 512)
    parser.add_argument('--lstm_hidden_size', type = int, default = 512)
    parser.add_argument('--window_size', type = int, default = 5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--youtube8m_feat_dim', type=int, default=2048)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--stft_feat_dim', type=int, default=512)
    args = parser.parse_args()
    from mmcv import Config
    model = LGSS(args)
    youtube8m_feat = torch.randn(args.batch_size, args.window_size * 2, args.youtube8m_feat_dim)
    stft_feat = torch.randn(args.batch_size, args.window_size * 2, 257, 90)
    output = model(youtube8m_feat, stft_feat)
    print(output.data.size())
