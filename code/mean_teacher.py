from resnet_model import create_net
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, ReLU, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve as pr_curve
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import resnet
from plot import plot_pr, test_plot_pr, kfold_plot_pr
from dataloader import mean_teacher_train_dataloader, mean_teacher_unlabeled_dataloader, dataloader, fetch_filenames, fetch_unlabeled_filenames, train_val_split, batch_augment
import matplotlib.pyplot as plt
from mixmatch import ema
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


def linear_rampup(epoch, rampup_length=16):
    if rampup_length == 0:
        return 1.
    else:
        rampup = tf.clip_by_value(epoch / rampup_length, 0., 1.)
        return float(rampup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--dataset', type=str, default='TrainingV2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unlabeled_batch_size', type=int, default=32)
    parser.add_argument('--val_iterations', type=int, default=437)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_parallel_calls', type=int, default=4)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--rampup_length', type=int, default=16)
    parser.add_argument('--resnet_ver', type=int, default=3)
    parser.add_argument('--resnet_n', type=int, default=2)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='ema decay for ema model vars (default: 0.999)')
    parser.add_argument('--pr_curve_file', type=bool, default=True)
    args = parser.parse_args()

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
    logging.info('dataset: {}'.format(args.dataset))
    logging.info('epochs: {}'.format(args.epochs))
    logging.info('batch_size: {}'.format(args.batch_size))
    logging.info('unlabeled_batch_size: {}'.format(args.unlabeled_batch_size))
    logging.info('val_iterations: {}'.format(args.val_iterations))
    logging.info('shuffle: {}'.format(args.shuffle))
    logging.info('num_parallel_calls: {}'.format(args.num_parallel_calls))
    logging.info('buffer_size: {}'.format(args.buffer_size))
    logging.info('prefetch: {}'.format(args.prefetch))
    logging.info('learning_rate: {}'.format(args.learning_rate))
    logging.info('beta_1: {}'.format(args.beta_1))
    logging.info('beta_2: {}'.format(args.beta_2))
    logging.info('lambda_u: {}'.format(args.lambda_u))
    logging.info('rampup_length: {}'.format(args.rampup_length))
    logging.info('resnet_ver: {}'.format(args.resnet_ver))
    logging.info('resnet_n: {}'.format(args.resnet_n))
    logging.info('ema-decay: {}'.format(args.ema_decay))

    ###################################
    ## GRABBING FILENAMES FOR IMAGES ##
    ###################################
    train_filenames = fetch_filenames(
        'data/{dataset}/DLS'.format(dataset=args.dataset), 'train')
    train_filenames, val_filenames = train_val_split(train_filenames)
    unlabeled_filenames = fetch_unlabeled_filenames(
        'data/{dataset}/DLS'.format(dataset=args.dataset), 'unlabeled')
    test_filenames = fetch_filenames('data/{dataset}/DLS'.format(dataset=args.dataset), 'test')

    ##########################################################
    ## MAKING DATALOADERS FOR TRAINING, VALIDATION, TESTING ##
    ##########################################################
    train_dataloader = mean_teacher_train_dataloader(
        filenames=train_filenames,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_parallel_calls=args.num_parallel_calls,
        buffer_size=args.buffer_size,
        prefetch=args.prefetch,
    )
    unlabeled_dataloader = mean_teacher_unlabeled_dataloader(
        filenames=unlabeled_filenames,
        batch_size=args.unlabeled_batch_size,
        shuffle=args.shuffle,
        num_parallel_calls=args.num_parallel_calls,
        buffer_size=args.buffer_size,
        prefetch=args.prefetch,
        drop_remainder=True,
    )
    val_dataloader = dataloader(
        filenames=val_filenames,
        batch_size=64,
        shuffle=False,
        num_parallel_calls=args.num_parallel_calls,
        buffer_size=1,
        prefetch=1,
    )
    test_dataloader = dataloader(
        filenames=test_filenames,
        batch_size=1,
        shuffle=False,
        num_parallel_calls=args.num_parallel_calls,
        buffer_size=1,
        prefetch=1,
    )
    #######################################
    ## INITIALIZE MODEL, LOSSES, METRICS ##
    #######################################
    model = create_net('model', args.resnet_ver, args.resnet_n)
    ema_model = create_net('model', args.resnet_ver, args.resnet_n)
    copy_model(model, ema_model)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    test_precision = tf.keras.metrics.Precision(name='test_precision')
    test_recall = tf.keras.metrics.Recall(name='test_recall')
    test_pr_auc = tf.keras.metrics.AUC(name='test_pr_auc', curve='PR')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
    )

    model.summary(print_fn=logging.info)
    best_accuracy = float('-inf')

    ###################
    ## TRAINING LOOP ##
    ###################
    for epoch in range(args.epochs):
        @tf.function
        def train_step(model, ema_model, batchX, batchU1, batchU2, lambda_u, ema_decay):
            with tf.GradientTape() as tape:
                logitsX = model(batchX[0])
                logitsU1 = model(batchU1)
                predictionsU1 = tf.nn.softmax(logitsU1)
                logitsU2 = model(batchU2)
                predictionsU2 = tf.nn.softmax(logitsU2)

                labelsX_one_hot = tf.squeeze(
                    tf.one_hot(batchX[1], 2, dtype=tf.int32))
                lossX = loss_object(labelsX_one_hot, logitsX)
                lossU = tf.reduce_mean(
                    (predictionsU1-predictionsU2)**2)
                lossU = tf.reduce_mean(lossU)
                loss = lossX + lambda_u*lossU
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            ema(model, ema_model, ema_decay)

            predictionsX = tf.argmax(logitsX, axis=1, output_type=tf.int32)
            train_loss(loss)
            train_accuracy(batchX[1], predictionsX)

        @tf.function
        def val_step(model, images, labels):
            logits = model(images)
            labels_one_hot = tf.squeeze(tf.one_hot(labels, 2, dtype=tf.int32))
            v_loss = loss_object(labels_one_hot, logits)

            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            val_loss(v_loss)
            val_accuracy(labels, predictions)

        iteratorX = iter(train_dataloader)
        iteratorU = iter(unlabeled_dataloader)

        train_start = time.time()

        ema_decay = tf.constant(args.ema_decay)
        for batch_num in range(args.val_iterations):
            lambda_u = args.lambda_u * linear_rampup(
                epoch + batch_num/args.val_iterations, args.rampup_length)
            lambda_u = tf.constant(lambda_u)
            try:
                batchX = next(iteratorX)
            except:
                iteratorX = iter(train_dataloader)
                batchX = next(iteratorX)
            try:
                batchU = next(iteratorU)
            except:
                iteratorU = iter(unlabeled_dataloader)
                batchU = next(iteratorU)

            batchU1 = tf.concat([batchX[0], batchU[0]], axis=0)
            batchU2 = tf.concat([batchX[1], batchU[1]], axis=0)
            batchX = (batchX[0], batchX[2])

            train_step(model, ema_model, batchX, batchU1,
                       batchU2, lambda_u, ema_decay)

        train_end = time.time()

        val_start = time.time()
        for images, labels in val_dataloader:
            val_step(model, images, labels)
        val_end = time.time()

        template = 'Epoch: {}, Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}, Train Elapsed Time: {}, Validation Elapsed Time: {}'
        logging.info(template.format(
            epoch,
            train_loss.result(),
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

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

    ###############
    ## TEST LOOP ##
    ###############
    model = create_net('model', args.resnet_ver, args.resnet_n)
    model.load_weights(os.path.join(model_path, 'ema_model.ckpt'))

    @tf.function
    def test_step(model, images, labels):
        logits = model(images)
        labels_one_hot = tf.expand_dims(tf.squeeze(
            tf.one_hot(labels, 2, dtype=tf.int32)), axis=0)
        t_loss = loss_object(labels_one_hot, logits)

        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        test_precision(labels, predictions)
        test_recall(labels, predictions)
        test_pr_auc(labels, predictions)

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

    if args.pr_curve_file:
        test_plot_pr(model, os.path.join(
            model_path, 'test_plot_pr'), test_dataloader)
