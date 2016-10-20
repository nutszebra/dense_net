# What's this
Implementation of Densely Connected Convolutional Networks by chainer  

# Dependencies

    git clone https://github.com/nutszebra/dense_net.git
    cd dense_net
    git clone https://github.com/nutszebra/trainer.git

# How to run
    python main.py -p ./ -e 300 -b 64 -g 0 -s 1 -trb 4 -teb 4 -lr 0.1

# Details about my implementation
main.py executes dense_net on cifar10.  
The depth of dense_net is 100, due to the lack of gpu memory.  
Even though the way of data-augmentation is slightly different, I could confirmed almost same accuracy that is reported by [[1]][Paper]

# References
Densely Connected Convolutional Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1608.06993 "Paper"
