### DA-LSTM

This is an implementation of paper ["A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"](https://arxiv.org/pdf/1704.02971). I only did a test to predict the price of AAPL.US by its historical data as well as the price of its opponent MSFT.US. 

#### Dataset
Downloaded from [NASDAQ 100 STOCK DATA](http://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html).

#### Argument

`-e`, `--epoch` - the number of epochs

`-b`, `--batch` - the batch size

`-s`, `--split` - the split ratio of train and test set

`-i`, `--interval` - save models every interval epochs

`-l`, `--lrate` - learning rate of optimizor

`-t`, `—test` - test phase

`-m`, `—model` - if in test phase, the models name(if model name is "encoder50" and decoder50", inptut 50)

#### Sample train

Traing 500 epochs, with batch-size 128, save models every 100 epochs.

```
Python3 trainer -e 500 -b 128 -i 100
```

#### Sample test

Test data use model "encoder50" and "decoder50"

```
Python3 trainer -t -m 50
```

