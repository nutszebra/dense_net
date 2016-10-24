# What's this
Implementation of Densely Connected Convolutional Networks (DCCN) by chainer  

# Dependencies

    git clone https://github.com/nutszebra/dense_net.git
    cd dense_net
    git clone https://github.com/nutszebra/trainer.git

# How to run
    python main.py -p ./ -e 300 -b 64 -g 0 -s 1 -trb 4 -teb 4 -lr 0.1

# Details about my implementation
main.py runs DCCN on cifar10.  
The depth of DCCN is 40, due to the lack of gpu memory.  
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Pictures are randomly resized in the range of [28, 36], then 26x26 patches are extracted randomly.
Horizontal flipping is applied with 0.5 probability.  

# Result
As a result, I could confirm 95.12% total accuracy at epoch 251 and this result has almost the same accuracy as reported by [[1]][Paper].  

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
