# What's this
Implementation of Densely Connected Convolutional Networks (DCCN) by chainer  

# Dependencies

    git clone https://github.com/nutszebra/dense_net.git
    cd dense_net
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -e 300 -b 64 -g 0 -s 1 -trb 4 -teb 4 -lr 0.1

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

# Cifar10 result
As a result, I could confirm 95.12% total accuracy at epoch 250 and this result has almost the same accuracy as reported by [[1]][Paper].  

| network           | depth | k  | total accuracy (%) |
|:------------------|-------|----|-------------------:|
| DCCN [[1]][Paper] | 40    | 12 | 94.76              |
| my implementation | 40    | 12 | 95.12              |
| DCCN [[1]][Paper] | 100   | 12 | 95.9               |
| DCCN [[1]][Paper] | 100   | 24 | 96.26              |

<img src="https://github.com/nutszebra/dense_net/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/dense_net/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">
# References
Densely Connected Convolutional Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1608.06993 "Paper"
