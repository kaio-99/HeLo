import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        
        self.parser.add_argument('--root_path', type=str, default='..\Features\\', help='the root path of dataset')
        self.parser.add_argument('--train_size', type=int, default=0.8, help='ratio of training set')
        self.parser.add_argument('--n_channels', type=int, default=18, help='number of the EEG channels')
        self.parser.add_argument('--dim_eeg', type=int, default=5, help='feature dimension of EEG')
        self.parser.add_argument('--dim_gsr', type=int, default=28, help='feature dimension of GSR')
        self.parser.add_argument('--dim_ppg', type=int, default=27, help='feature dimension of PPG')
        self.parser.add_argument('--dim_video', type=int, default=768, help='feature dimension of video')
        self.parser.add_argument('--depth', type=int, default=1, help='depth of the transformer blocks')
        self.parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
        self.parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer dimension of ffn')
        self.parser.add_argument('--heads', type=int, default=4, help='number of the attention heads')
        self.parser.add_argument('--heads_lcd', type=int, default=1, help='number of the attention heads in lcdca')
        self.parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')
        self.parser.add_argument('--emotion_classes', type=int, default=10, help='number of the emotion classes')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train [default: 200]')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--sub_list', type=list, default=[1,2,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80], help='subject id list')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

