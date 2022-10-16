from resnet_model import create_net_mixmatch as create_net
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, ReLU, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve as pr_curve
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import resnet
from plot import plot_pr, test_plot_pr
from dataloader import train_dataloader, mixmatch_train_dataloader, unlabeled_dataloader, dataloader, fetch_filenames, fetch_unlabeled_filenames, train_val_split
from mixmatch import mixmatch, semi_loss, linear_rampup, interleave, weight_decay, ema
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import os
import glob
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
sys.path.append('..')


def copy_model(model_A, model_B):
    for a, b in zip(model_A.trainable_variables, model_B.trainable_variables):
        a.assign(tf.identity(b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--dataset', type=str, default='TrainingV2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_parallel_calls', type=int, default=4)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--resnet_ver', type=int, default=3)
    parser.add_argument('--resnet_n', type=int, default=2)
    # assuming a batch size of 32
    parser.add_argument('--val_iteration', type=int, default=437)
    parser.add_argument('--T', type=float, default=0.5,
                        help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=2,
                        help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda_u', type=float, default=100,
                        help='multiplier for unlabelled loss (default: 100)')
    parser.add_argument('--rampup_length', type=int, default=16,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='ema decay for ema model vars (default: 0.999)')
    parser.add_argument('--pr_curve_file', type=bool, default=True)
    args = parser.parse_args()
    args = vars(args)

    ###################################
    ## CREATE DIRECTORIES FOR MODELS ##
    ###################################
    version = 1
    template = os.path.join(os.getcwd(), 'saved_models',
                            'ResNet11', 'version {}')
    model_path = template.format(version)
    while os.path.exists(model_path):
        version += 1
        model_path = template.format(version)
    os.makedirs(model_path)

    fh = logging.FileHandler(os.path.join(model_path, 'results.log'), mode='w')
    logging.getLogger().addHandler(fh)

    ################################
    ## LOGGING PARAMETERS FOR RUN ##
    ################################
    logging.info('dataset: {}'.format(args['dataset']))
    logging.info('epochs: {}'.format(args['epochs']))
    logging.info('batch_size: {}'.format(args['batch_size']))
    logging.info('shuffle: {}'.format(args['shuffle']))
    logging.info('num_parallel_calls: {}'.format(args['num_parallel_calls']))
    logging.info('buffer_size: {}'.format(args['buffer_size']))
    logging.info('prefetch: {}'.format(args['prefetch']))
    logging.info('learning_rate: {}'.format(args['learning_rate']))
    logging.info('beta_1: {}'.format(args['beta_1']))
    logging.info('beta_2: {}'.format(args['beta_2']))
    logging.info('resnet_ver: {}'.format(args['resnet_ver']))
    logging.info('resnet_n: {}'.format(args['resnet_n']))
    logging.info('val_iteration: {}'.format(args['val_iteration']))
    logging.info('T: {}'.format(args['T']))
    logging.info('K: {}'.format(args['K']))
    logging.info('alpha: {}'.format(args['alpha']))
    logging.info('lambda-u: {}'.format(args['lambda_u']))
    logging.info('rampup-length: {}'.format(args['rampup_length']))
    logging.info('weight-decay: {}'.format(args['weight_decay']))
    logging.info('ema-decay: {}'.format(args['ema_decay']))

    ###################################
    ## GRABBING FILENAMES FOR IMAGES ##
    ###################################
    train_filenames = fetch_filenames('data/{dataset}/DLS'.format(dataset=args['dataset']), 'train')
    train_filenames, val_filenames = train_val_split(train_filenames)
    unlabeled_filenames = fetch_unlabeled_filenames(
        'data/{dataset}/DLS'.format(dataset=args['dataset']), 'unlabeled')
    test_filenames = fetch_filenames('data/{dataset}/DLS'.format(dataset=args['dataset']), 'test')

    ##########################################################
    ## MAKING DATALOADERS FOR TRAINING, VALIDATION, TESTING ##
    ##########################################################
    train_dataloader = mixmatch_train_dataloader(
        filenames=train_filenames,
        batch_size=args['batch_size'],
        shuffle=args['shuffle'],
        num_parallel_calls=args['num_parallel_calls'],
        buffer_size=args['buffer_size'],
        prefetch=args['prefetch'],
        drop_remainder=True,
    )
    unlabeled_dataloader = unlabeled_dataloader(
        filenames=unlabeled_filenames,
        batch_size=args['batch_size'],
        shuffle=args['shuffle'],
        num_parallel_calls=args['num_parallel_calls'],
        buffer_size=args['buffer_size'],
        prefetch=args['prefetch'],
        drop_remainder=True,
    )
    val_dataloader = dataloader(
        filenames=val_filenames,
        batch_size=64,
        shuffle=False,
        num_parallel_calls=args['num_parallel_calls'],
        buffer_size=1,
        prefetch=1,
    )
    test_dataloader = dataloader(
        filenames=test_filenames,
        batch_size=1,
        shuffle=False,
        num_parallel_calls=args['num_parallel_calls'],
        buffer_size=1,
        prefetch=1,
    )

    #######################################
    ## INITIALIZE MODEL, LOSSES, METRICS ##
    #######################################
    model = create_net('model', args['resnet_ver'], args['resnet_n'])
    ema_model = create_net('model', args['resnet_ver'], args['resnet_n'])
    copy_model(model, ema_model)

    #loss_object = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    xe_loss_avg = tf.keras.metrics.Mean(name='xe_loss')
    l2u_loss_avg = tf.keras.metrics.Mean(name='l2u_loss')
    total_loss_avg = tf.keras.metrics.Mean(name='total_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    test_precision = tf.keras.metrics.Precision(name='test_precision')
    test_recall = tf.keras.metrics.Recall(name='test_recall')
    test_pr_auc = tf.keras.metrics.AUC(name='test_pr_auc', curve='PR')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args['learning_rate'],
        beta_1=args['beta_1'],
        beta_2=args['beta_2'],
    )

    model.summary(print_fn=logging.info)
    best_accuracy = float('-inf')

    ###################
    ## TRAINING LOOP ##
    ###################
    for epoch in range(args['epochs']):
        @tf.function
        def train_step(model, ema_model, batchX, batchU):
            with tf.GradientTape() as tape:
                # run mixmatch
                XU, XUy = mixmatch(
                    model, batchX[0], batchX[1], batchU, args['T'], args['K'], args['beta'])
                logits = [model(XU[0])]
                for batch in XU[1:]:
                    logits.append(model(batch))
                logits = interleave(logits, args['batch_size'])
                logits_x = logits[0]
                logits_u = tf.concat(logits[1:], axis=0)

                # compute loss
                xe_loss, l2u_loss = semi_loss(
                    XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
                total_loss = xe_loss + lambda_u * l2u_loss

            # compute gradients and run optimizer step
            grads = tape.gradient(
                total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
            ema(model, ema_model, args['ema_decay'])
            weight_decay(
                model=model, decay_rate=args['weight_decay'] * args['learning_rate'])

            xe_loss_avg(xe_loss)
            l2u_loss_avg(l2u_loss)
            total_loss_avg(total_loss)
            train_accuracy(tf.argmax(batchX[1], axis=1, output_type=tf.int32),
                           tf.argmax(model(tf.cast(batchX[0], dtype=tf.float32), training=False), axis=1, output_type=tf.int32))

        @tf.function
        def val_step(model, images, labels):
            logits = model(images, training=False)

            xe_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels, 2, dtype=tf.int32), logits=logits)
            val_loss(xe_loss)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            val_accuracy(tf.squeeze(labels), prediction)

        train_start = time.time()

        datasetX = train_dataloader
        datasetU = unlabeled_dataloader

        iteratorX = iter(datasetX)
        iteratorU = iter(datasetU)

        for batch_num in range(args['val_iteration']):
            lambda_u = args['lambda_u'] * linear_rampup(
                epoch + batch_num/args['val_iteration'], args['rampup_length'])
            try:
                batchX = next(iteratorX)
            except:
                iteratorX = iter(datasetX)
                batchX = next(iteratorX)
            try:
                batchU = next(iteratorU)
            except:
                iteratorU = iter(datasetU)
                batchU = next(iteratorU)

            # args['beta'].assign(np.random.beta(
            #    args['alpha'], args['alpha']))
            args['beta'] = tfp.distributions.Beta(
                args['alpha'], args['alpha']).sample()

            train_step(model, ema_model, batchX, batchU)

        train_end = time.time()

        val_start = time.time()
        for images, labels in val_dataloader:
            val_step(ema_model, images, labels)
        val_end = time.time()

        template = 'Epoch: {}, Train Loss: {}, X Loss: {}, U Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Train Elapsed Time: {}, Validation Elapsed Time: {}'
        logging.info(template.format(
            epoch,
            total_loss_avg.result(),
            xe_loss_avg.result(),
            l2u_loss_avg.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100,
            train_end-train_start,
            val_end-val_start)
        )

        if val_accuracy.result() > best_accuracy:
            best_accuracy = val_accuracy.result()
            ema_model.save_weights(os.path.join(
                model_path, 'ema_model.ckpt'))

        if train_accuracy.result() == 1:
            break

        xe_loss_avg.reset_states()
        l2u_loss_avg.reset_states()
        total_loss_avg.reset_states()
        train_accuracy.reset_states()

        val_loss.reset_states()
        val_accuracy.reset_states()

    ###############
    ## TEST LOOP ##
    ###############
    test_predictions = []
    test_labels = []
    model = create_net('model', args['resnet_ver'], args['resnet_n'])
    model.load_weights(os.path.join(model_path, 'ema_model.ckpt'))

    @tf.function
    def test_step(model, images, labels):
        logits = model(images, training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, 2, dtype=tf.int32), logits=logits)
        test_loss(xe_loss)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(predictions, labels[0])
        test_precision(predictions, labels[0])
        test_recall(predictions, labels[0])
        test_pr_auc(predictions, labels[0])

    for images, labels in test_dataloader:
        test_step(model, images, labels)

    template = 'Test Accuracy: {}, Test Loss: {}, Test Precision: {}, Test Recall {}, Test PR: {}'
    logging.info(template.format(
        test_accuracy.result()*100,
        test_loss.result(),
        test_precision.result(),
        test_recall.result(),
        test_pr_auc.result()
    )
    )

    if args['pr_curve_file']:
        test_plot_pr(model, os.path.join(
            model_path, 'test_plot_pr'), test_dataloader, output_dim=2)
