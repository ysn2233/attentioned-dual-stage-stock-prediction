import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from model import AttnEncoder, AttnDecoder
from dataset import Dataset
from torch import optim
import config

class Trainer:

    def __init__(self, driving, target, time_step, split, lr):
        self.dataset = Dataset(driving, target, time_step, split)
        self.encoder = AttnEncoder(input_size=self.dataset.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=time_step)
        self.decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=time_step)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.MSELoss()
        self.train_size, self.test_size = self.dataset.get_size()

    def train_minibatch(self, num_epochs, batch_size, interval):
        x_train, y_train, y_seq_train = self.dataset.get_train_set()
        for epoch in range(num_epochs):
            i = 0
            loss_sum = 0
            while (i < self.train_size):
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                batch_end = i + batch_size
                if (batch_end >= self.train_size):
                    batch_end = self.train_size
                var_x = self.to_variable(x_train[i: batch_end])
                var_y = self.to_variable(y_train[i: batch_end])
                var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                code = self.encoder(var_x)
                y_res = self.decoder(code, var_y_seq)
                loss = self.loss_func(y_res, var_y)
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                # print('[%d], loss is %f' % (epoch, 10000 * loss.data[0]))
                loss_sum += loss.data[0]
                i = batch_end
            print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))
            if (epoch + 1) % (interval) == 0 or epoch + 1 == num_epochs:
                torch.save(self.encoder.state_dict(), 'models/encoder' + str(epoch + 1) + '-norm' + '.model')
                torch.save(self.decoder.state_dict(), 'models/decoder' + str(epoch + 1) + '-norm' + '.model')

    def test(self, num_epochs, batch_size):
        x_train, y_train, y_seq_train = self.dataset.get_train_set()
        x_test, y_test, y_seq_test = self.dataset.get_test_set()
        y_pred_train = self.predict(x_train, y_train, y_seq_train, batch_size)
        y_pred_test = self.predict(x_test, y_test, y_seq_test, batch_size)
        plt.figure(figsize=(8,6), dpi=100)
        plt.plot(range(2000, self.train_size), y_train[2000:], label='train truth', color='black')
        plt.plot(range(self.train_size, self.train_size + self.test_size), y_test, label='ground truth', color='black')
        plt.plot(range(2000, self.train_size), y_pred_train[2000:], label='predicted train', color='red')
        plt.plot(range(self.train_size, self.train_size + self.test_size), y_pred_test, label='predicted test', color='blue')
        plt.xlabel('Days')
        plt.ylabel('Stock price of AAPL.US(USD)')
        plt.savefig('results/res-' + str(num_epochs) +'-' + str(batch_size) + '.png')


    def predict(self, x, y, y_seq, batch_size):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                batch_end = x.shape[0]
            var_x_input = self.to_variable(x[i: batch_end])
            var_y_input = self.to_variable(y_seq[i: batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            code = self.encoder(var_x_input)
            y_res = self.decoder(code, var_y_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i, -1]
            i = batch_end
        return y_pred


    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())



def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=1,
        help='the mini-batch size')
    parser.add_argument(
        '-s', '--split', type=float, default=0.8,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    mname = args.model
    trainer = Trainer(config.DRIVING, config.TARGET, 10, split, lr)
    if not test:
        trainer.train_minibatch(num_epochs, batch_size, interval)
    else:
        encoder_name = 'models/encoder' + mname + '.model'
        decoder_name = 'models/decoder' + mname + '.model'
        trainer.load_model(encoder_name, decoder_name)
        trainer.test(mname, batch_size)