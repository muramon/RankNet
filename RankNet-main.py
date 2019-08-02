import datetime
import torch
import torch.nn as nn
from train_test import train_step, test_step
from torch.utils.tensorboard import SummaryWriter
from design_model import design_model
from model import RankNet
from dataloader import L2RDataset


if __name__ == '__main__':

    for d in [32, 64, 128]:
        now = datetime.datetime.now()
        now = "{0:%Y%m%d%H%M}".format(now)
        w = SummaryWriter(log_dir="./logs/log-Dsize-{}-{}".format(d, now))
        file = open('./results/result-Dsize-{}-{}.txt'.format(d, now), 'a')
        total_ndcg = {}
        for k in [1, 3, 5, 10]:
            total_ndcg[k] = 0
        total_loss = 0
        dims = [46, 128, d, 32]
        models = {}
        actf1 = nn.ReLU()
        actf2 = nn.Sigmoid()
        max_epoch = 50
        file.write("epoch: {}".format(max_epoch))
        file.write('\n')

        for n in ['1', '2', '3', '4', '5']:
            layers = design_model(actf1, dims)
            models[n] = RankNet(layers)
            optimizer = torch.optim.Adam(models[n].parameters(), lr=0.001)
            train_file = './MQ2007/Fold%s/train.txt' % n
            val_file = './MQ2007/Fold%s/vali.txt' % n
            test_file = './MQ2007/Fold%s/test.txt' % n
            train_ds = L2RDataset(file=train_file, data_id='MQ2007_Super')
            val_ds = L2RDataset(file=val_file, data_id='MQ2007_Super')
            test_ds = L2RDataset(file=test_file, data_id='MQ2007_Super')
            best_val_ndcg_score = 0
            if n == '1':
                model_s = str(models[n])
                file.write(model_s)

            for epoch in range(max_epoch):
                epoch_train_loss = train_step(models[n], train_ds, optimizer)
                print("Epoch: {} Train Loss: {}".format(epoch, epoch_train_loss))
                epoch_train_dcg = test_step(models[n], train_ds)
                for k in [1, 3, 5, 10]:
                    print("Epoch: {} Train nDCG@{}: {}".format(epoch, k, epoch_train_dcg[k]))
                    w.add_scalar("train nDCG@%d %s" % (k, n), epoch_train_dcg[k], epoch)
                w.add_scalar("train loss %s" % n, epoch_train_loss, epoch)
                epoch_val_dcg = test_step(models[n], val_ds)
                for k in [1, 3, 5, 10]:
                    print("Epoch: {} Val nDCG@{}: {}".format(epoch, k, epoch_val_dcg[k]))
                    w.add_scalar("val nDCG@%d %s" % (k, n), epoch_val_dcg[k], epoch)
                if epoch_val_dcg[1] > best_val_ndcg_score:
                    best_epoch = epoch
                    best_loss = epoch_train_loss
                    best_val_ndcg_score = epoch_val_dcg[1]
                    torch.save(models[n], './models/relu/Fold%s' % n)
                print("--" * 50)

            val_model = torch.load('./models/relu/Fold%s' % n)
            test_ndcg = test_step(val_model, test_ds)
            for k in [1, 3, 5, 10]:
                total_ndcg[k] += test_ndcg[k]
                print("--" * 50)
                print("Test NDCG@{}: {}".format(k, test_ndcg[k]))
                print("--" * 50)
                file.write("Folder: {} Test NDCG@{}: {}".format(n, k, test_ndcg[k]))
                file.write('\n')
            file.write("Best epoch : {}".format(best_epoch))
            file.write('\n')
            file.write("Best train loss : {}".format(best_loss))
            file.write('\n')
            total_loss += best_loss
        for k in [1, 3, 5, 10]:
            ave_ndcg = total_ndcg[k] / 5
            print("Ave Test NDCG@{}: {}".format(k, ave_ndcg))
            file.write("Ave Test NDCG@{}: {}".format(k, ave_ndcg))
            file.write('\n')
        ave_loss = total_loss / 5
        file.write("Ave train loss in best : {}".format(ave_loss))

        w.close()
