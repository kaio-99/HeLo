import numpy as np
import scipy.io as sio
from torch.utils.data.dataset import Dataset

class DMERDataset(Dataset):
    def __init__(self, path='1.mat'):
        super(DMERDataset, self).__init__()

        self.path = path
        self.data = sio.loadmat(self.path)
        self.eeg = self.data['feas'][:, 0:90] 
        self.gsr = self.data['feas'][:, 90:118]
        self.ppg = self.data['feas'][:, 118:145]
        self.video = self.data['feas'][:, 145:]
        self.label = np.array([self.data['dis_label'][int(x)] for x in self.data["vids"].T[0]])
        self.label = self.label / self.label.sum(axis=1, keepdims=True)

    def __getitem__(self, index):
        eeg = self.eeg[index]
        gsr = self.gsr[index]
        ppg = self.ppg[index]
        video = self.video[index]
        label = self.label[index]
        return eeg, gsr, ppg, video, label
    
    def __len__(self):
        return len(self.label)