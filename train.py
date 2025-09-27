import torch
import torch.nn as nn
from params import Config
from utils import SetSeeds
from models.model import HeLo
from dataset import DMERDataset
from train_validate import train
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    seed = 3407
    print('seed is {}'.format(seed))
    print('training on:', device)
    SetSeeds(seed)

    args = Config().parse()

    sub_list = args.sub_list
    for sub in sub_list:
        print('Subject {}'.format(sub))
        dataset = DMERDataset(args.root_path + str(sub) + '.mat')
        train_size = int(args.train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = HeLo(args).to(device)

        criterionKL = nn.KLDivLoss(size_average=False)
        criterionMSE = torch.nn.MSELoss()  
        criterion = [criterionKL, criterionMSE]

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        train_losses = train(model, optimizer, scheduler, criterion, 
                                      train_loader, test_loader, args.epochs, sub)

        print('Finish Training of Subject {}'.format(sub))
