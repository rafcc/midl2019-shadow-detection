import chainer
from chainer import training
from chainer.training import extensions
import argparse
import time
import os
import random
import json
import numpy as np
import cupy
from PIL import Image

import nn
import datasets


@training.make_extension()
def output_image(trainer, prefix=''):
    model = trainer.updater.get_optimizer('main').target
    xp = chainer.backends.cuda.get_array_module(model.x)

    # Take the first image within the batch
    x = model.x[0][0]
    shadow = model.shadow[0][0]
    a_shadow = model.a_shadow[0][0]
    structure = model.structure[0][0]
    reconstruction = model.reconstruction[0][0]
    if xp.__name__ == 'cupy':
        x = cupy.asnumpy(x)
        a_shadow = cupy.asnumpy(a_shadow)
        shadow = cupy.asnumpy(shadow)
        structure = cupy.asnumpy(structure)
        reconstruction = cupy.asnumpy(reconstruction)

    # Save image
    image = np.hstack([x, a_shadow, shadow, structure, reconstruction]) * 255
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image.save(
        os.path.join(
            trainer.out, '{}img_iter{}.png'.format(prefix,
                                                   trainer.updater.iteration)))


@training.make_extension()
def progress_spin(trainer):
    s = ['|', '/', '-', '\\']
    print(s[trainer.updater.iteration % len(s)], end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', type=int, default=0, help='GPU to use. Defaults to 0.')
    parser.add_argument(
        '--batchsize',
        type=int,
        default=32,
        help='Minibatch size. Defaults to 32.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='Number of training epochs. Defaults to 10.')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Directory to output result. Defaults to `result_YYMMDD-HHMMSS`.')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./dataset',
        help='Root directory for dataset. Defaults to `./dataset`.')
    parser.add_argument(
        '--dataset_list',
        type=str,
        default='./dataset/list.txt',
        help=('List of images in the dataset. '
              'Defaults to `./dataset/list.txt`.'))
    parser.add_argument(
        '--test_list',
        type=str,
        default=None,
        help=('List of images in for test. If None, train set will be split. '
              'Defaults to None.'))
    parser.add_argument(
        '--resume', type=str, default=None, help='Snapshot path for resume.')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='Optimizer. Defaults to `Adam`.')
    parser.add_argument(
        '--alpha', type=float, default=1e-5, help='Alpha for adam.')
    parser.add_argument(
        '--lambda_l2', type=float, default=1., help='Weight for l2 loss.')
    parser.add_argument(
        '--lambda_l1', type=float, default=0., help='Weight for l1 loss.')
    parser.add_argument(
        '--lambda_ss',
        type=float,
        default=10.,
        help='Weight for augmented shadow loss.')
    parser.add_argument(
        '--lambda_ssreg',
        type=float,
        default=0.01,
        help='Weight for shadow regularization loss.')
    parser.add_argument(
        '--ssreg_thresh',
        type=float,
        default=1,
        help='Threshold for shadow regularization.')
    parser.add_argument(
        '--lambda_beta_shadow',
        type=float,
        default=0,
        help='Weight for beta distribution loss for shadow.')
    parser.add_argument(
        '--lambda_beta_structure',
        type=float,
        default=1e-8,
        help='Weight for beta distribution loss for structure.')
    parser.add_argument(
        '--lambda_edge',
        type=float,
        default=0,
        help='Weight for edge loss.')
    args = parser.parse_args()

    # Output directory
    if args.out is None:
        args.out = 'result_' + time.strftime('%y%m%d-%H%M%S')
    os.makedirs(args.out, exist_ok=True)

    # Save args
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Prepare list of data
    with open(args.dataset_list) as f:
        data_list = f.readlines()
    data_list = [p.strip() for p in data_list]
    if args.test_list is not None:
        with open(args.test_list) as f:
            test_data_list = f.readlines()
        test_data_list = [p.strip() for p in test_data_list]
    else:
        random.seed(0)
        random.shuffle(list(range(len(data_list))))
        train_data_list = data_list[1000:]
        test_data_list = data_list[:1000]

    # Set up dataset
    train = datasets.AugmentedImageDataset(
        paths=train_data_list, root=args.dataset_root, train=True)
    test = datasets.AugmentedImageDataset(
        paths=test_data_list, root=args.dataset_root, train=False)

    # Set up iterator
    train_iter = chainer.iterators.MultithreadIterator(
        train, args.batchsize, n_threads=16)
    test_iter = chainer.iterators.MultithreadIterator(
        test, args.batchsize, repeat=False, shuffle=False, n_threads=16)

    # Set up models
    model = nn.ShadowSplitter(
        lambda_l2=args.lambda_l2,
        lambda_l1=args.lambda_l1,
        lambda_beta_shadow=args.lambda_beta_shadow,
        lambda_beta_structure=args.lambda_beta_structure,
        lambda_ss=args.lambda_ss,
        ssreg_thresh=args.ssreg_thresh,
        lambda_ssreg=args.lambda_ssreg,
        lambda_edge=args.lambda_edge,
    )
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_array(args.gpu).use()
        model.to_gpu()

    # Set up optimizer
    # optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer = getattr(chainer.optimizers, args.optimizer)
    optimizer = optimizer(args.alpha)
    optimizer.setup(model)

    # Set up trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    evaluator.name = 'val'  # Set shorter name
    trainer.extend(evaluator, trigger=(100, 'iteration'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    print_reports = ['main/loss']
    if args.lambda_l2 != 0:
        print_reports.append('main/l2_loss')
    if args.lambda_l1 != 0:
        print_reports.append('main/l1_loss')
    if args.lambda_beta_shadow != 0:
        print_reports.append('main/beta_shadow_loss')
    if args.lambda_beta_structure != 0:
        print_reports.append('main/beta_structure_loss')
    if args.lambda_ss != 0:
        print_reports.append('main/ss_loss')
    if args.lambda_ssreg != 0:
        print_reports.append('main/ssreg_loss')
    if args.lambda_edge != 0:
        print_reports.append('main/edge_loss')
    print_reports = print_reports + ['val/' + s for s in print_reports]
    print_reports = ['epoch', 'iteration'] + print_reports + ['elapsed_time']
    trainer.extend(
        extensions.PrintReport(print_reports),
        trigger=(100, 'iteration'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Output image (test)
    trainer.extend(
        lambda t: output_image(t, 'test'),
        trigger=(100, 'iteration'),
        priority=0)

    # Output image (train)
    trainer.extend(output_image, trigger=(100, 'iteration'), priority=1000)

    # Iteration progress
    trainer.extend(progress_spin, trigger=(1, 'iteration'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
