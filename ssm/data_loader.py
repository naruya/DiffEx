import os
import random
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir, T, H, mode):
        self.T = T
        self.H = H
        
        traj_path = os.path.join(data_dir, "traj_{}.pkl".format(mode))
        self.traj_a, self.traj_o = joblib.load(traj_path)

        self.a_mean, self.a_std, self.o_mean, self.o_std = \
            joblib.load(os.path.join(data_dir, "param.pkl"))

        self.traj_a = ((self.traj_a - self.a_mean) / self.a_std)
        self.traj_o = ((self.traj_o - self.o_mean) / self.o_std)

    def __len__(self):
        return len(self.traj_a)

    def __getitem__(self, idx):
        # TODO: 直前3フレームも使って、T+4 にする
        t = np.random.randint(self.H - (self.T+1))
        x = self.traj_o[idx, t  :t+self.T+1]
        a = self.traj_a[idx, t+1:t+self.T+1]
        x = np.transpose(x, [0,1])
        x_0, x = x[0], x[1:]
        return x_0, x, a


class MyDataLoader(DataLoader):
    def __init__(self, mode, args):
        self.mode = mode
        SEED = args.seed
        np.random.seed(SEED)

        dataset = MyDataset(
            data_dir=args.data_dir,
            T=args.T,
            H=args.H,
            mode=mode,
        )
        super(MyDataLoader, self).__init__(dataset,
                                           batch_size=args.B,
                                           shuffle=args.shuffle,
                                           drop_last=True,
                                           num_workers=4,
                                           pin_memory=True)