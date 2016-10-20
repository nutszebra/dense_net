import sys
sys.path.append('./trainer')
import argparse
import dense_net
import nutszebra_cifar10 as nutszebra_cifar10
import nutszebra_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--save_path', '-p',
                        default='/home/suguru/fashion/model/pants/vgg_A_relu_bn_after_data_augmentation/',
                        help='model and optimizer will be saved every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=300,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='leraning rate')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')

    print('generating model')
    model = dense_net.DenselyConnectedCNN(10, block_num=3, block_size=12, growth_rate=12)
    print('Done')
    optimizer = nutszebra_optimizer.OptimizerDense(model)
    args['model'] = model
    args['optimizer'] = optimizer
    main = nutszebra_cifar10.TrainCifar10(**args)
    main.run()
