import torch
from params import Config
from utils import SetSeeds
from models.model import HeLo
from dataset import DMERDataset
from train_validate import test
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    seed = 3407
    print('seed is {}'.format(seed))
    print('testing on:', device)
    SetSeeds(seed)

    args = Config().parse()
    
    metrics_all = []
    sub_list = args.sub_list
    for sub in sub_list:
        print('Subject {}'.format(sub))
        dataset = DMERDataset(args.root_path + str(sub) + '.mat')
        train_size = int(args.train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = HeLo(args).to(device)
        model.load_state_dict(torch.load('checkpoint/Sub_' + str(sub) + '_epoch_' + str(args.epochs - 1) + '.pth'))

        metrics = test(model, test_loader)
        metrics_all.append(metrics)

        print('Metrics of Subject {}: metrics: {}'.format(sub, metrics))
    
    print('Final_Metrics: {}'.format(sum(metrics_all) / len(sub_list)))
        