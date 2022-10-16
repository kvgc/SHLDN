from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, ReLU, Lambda, MaxPool2D, BatchNormalization
from tensorflow.keras import Model, Sequential
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
sys.path.append('..')

# Various APIs available for making models: sequential, functional, subclass.
# ResNet and everything we want done can be done with sequential and functional apis

########################
## MAKE YOUR NET HERE ##
########################


def ConvMeanPool(inputs, filters, kernel_size):
    INPUT = Input(shape=inputs.shape[1:])
    x = Conv2D(filters, kernel_size, padding='same')(INPUT)
    OUTPUT = (x[:, ::2, ::2, :] + x[:, ::2, 1::2, :] +
              x[:, 1::2, ::2, :] + x[:, 1::2, 1::2, :])/4
    return Model(inputs=INPUT, outputs=OUTPUT)


def MeanPoolConv(inputs, filters, kernel_size):
    INPUT = Input(shape=inputs.shape[1:])
    x = INPUT
    x = (x[:, ::2, ::2, :] + x[:, ::2, 1::2, :] +
         x[:, 1::2, ::2, :] + x[:, 1::2, 1::2, :])/4
    OUTPUT = Conv2D(filters, kernel_size, padding='same')(x)
    return Model(inputs=INPUT, outputs=OUTPUT)


def UpsampleConv(inputs, filters, kernel_size):
    INPUT = Input(shape=inputs.shape[1:])
    x = tf.concat([INPUT, INPUT, INPUT, INPUT], axis=-1)
    x = tf.nn.depth_to_space(x, 2)
    OUTPUT = Conv2D(filters, kernel_size, padding='same')(x)
    return Model(inputs=INPUT, outputs=OUTPUT)


def residual_block(name, inputs, kernel_size, strides, input_ch, output_ch, upsample=False):
    INPUT = Input(shape=inputs.shape[1:])
    if upsample == True:
        shortcut = UpsampleConv(inputs, output_ch, kernel_size)(INPUT)

        x = INPUT
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpsampleConv(inputs, output_ch, kernel_size)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        OUTPUT = Conv2D(output_ch, 3, padding='same')(x) + shortcut
        return Model(inputs=INPUT, outputs=OUTPUT, name=name)
    else:
        shortcut = MeanPoolConv(inputs, output_ch, kernel_size)(INPUT)

        x = INPUT
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(output_ch, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        OUTPUT = ConvMeanPool(x, output_ch, kernel_size)(x) + shortcut
        return Model(inputs=INPUT, outputs=OUTPUT, name=name)


def create_discriminator(z_dim, dim=10, name='Discriminator'):
    INPUT = Input(shape=[128, 128, 3])
    x = Conv2D(dim, 3, padding='same')(INPUT)
    x = residual_block('res1', x, 3, 1, input_ch=dim, output_ch=2*dim)(x)
    x = residual_block('res2', x, 3, 1, input_ch=2*dim, output_ch=4*dim)(x)
    x = residual_block('res3', x, 3, 1, input_ch=4*dim, output_ch=8*dim)(x)
    x = residual_block('res4', x, 3, 1, input_ch=8*dim, output_ch=16*dim)(x)
    x = residual_block('res5', x, 3, 1, input_ch=16*dim, output_ch=16*dim)(x)
    x = Flatten()(x)
    OUTPUT = Dense(1)(x)
    return Model(inputs=INPUT, outputs=OUTPUT)


def create_generator(z_dim, dim=10, name='Generator'):
    INPUT = Input((z_dim,))
    x = Dense(dim*4*4*16)(INPUT)
    x = tf.reshape(x, (-1, 4, 4, dim*16))
    # x = tf.compat.v1.placeholder_with_default(x, shape=[None, 8, 8, dim*16])
    x = residual_block('res1', x, 3, 1, input_ch=dim*16,
                       output_ch=dim*16, upsample=True)(x)
    x = residual_block('res2', x, 3, 1, input_ch=dim*16,
                       output_ch=dim*8, upsample=True)(x)
    x = residual_block('res3', x, 3, 1, input_ch=dim*8,
                       output_ch=dim*4, upsample=True)(x)
    x = residual_block('res4', x, 3, 1, input_ch=dim*4,
                       output_ch=dim*2, upsample=True)(x)
    x = residual_block('res5', x, 3, 1, input_ch=dim*2,
                       output_ch=dim*1, upsample=True)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    OUTPUT = Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    return Model(inputs=INPUT, outputs=OUTPUT, name=name)


def dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (128, 128))
        return image

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=lambda x: tf.py_function(
            transform, [x], [tf.float32]),
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size)

    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_dim', type=int, default=10)
    parser.add_argument('--d_dim', type=int, default=10)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--LAMBDA', type=float, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_parallel_calls', type=int, default=4)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--prefetch', type=int, default=1000)
    parser.add_argument('--G_learning_rate', type=float, default=0.001)
    parser.add_argument('--G_beta_1', type=float, default=0.0)
    parser.add_argument('--G_beta_2', type=float, default=0.999)
    parser.add_argument('--D_learning_rate', type=float, default=0.001)
    parser.add_argument('--D_beta_1', type=float, default=0.0)
    parser.add_argument('--D_beta_2', type=float, default=0.999)
    args = parser.parse_args()

    ###################################
    ## CREATE DIRECTORIES FOR MODELS ##
    ###################################
    version = 1
    model_template = os.path.join(os.getcwd(), 'saved_models',
                                  'WGAN-GP', 'version {}')
    image_template = os.path.join(os.getcwd(), 'generated_images',
                                  'WGAN-GP', 'version {}')
    model_path = model_template.format(version)
    image_path = image_template.format(version)
    while os.path.exists(model_path) or os.path.exists(image_path):
        version += 1
        model_path = model_template.format(version)
        image_path = image_template.format(version)
    os.makedirs(model_path)
    os.makedirs(image_path)

    fh = logging.FileHandler(os.path.join(model_path, 'results.log'), mode='w')
    logging.getLogger().addHandler(fh)

    ################################
    ## LOGGING PARAMETERS FOR RUN ##
    ################################
    logging.info('epochs: {}'.format(args.epochs))
    logging.info('batch_size: {}'.format(args.batch_size))
    logging.info('iterations: {}'.format(args.iterations))
    logging.info('z_dim: {}'.format(args.z_dim))
    logging.info('g_dim: {}'.format(args.g_dim))
    logging.info('d_dim: {}'.format(args.d_dim))
    logging.info('n_critic: {}'.format(args.n_critic))
    logging.info('LAMBDA: {}'.format(args.LAMBDA))
    logging.info('shuffle: {}'.format(args.shuffle))
    logging.info('num_parallel_calls: {}'.format(args.num_parallel_calls))
    logging.info('buffer_size: {}'.format(args.buffer_size))
    logging.info('prefetch: {}'.format(args.prefetch))
    logging.info('G_learning_rate: {}'.format(args.G_learning_rate))
    logging.info('G_beta_1: {}'.format(args.G_beta_1))
    logging.info('G_beta_2: {}'.format(args.G_beta_2))
    logging.info('D_learning_rate: {}'.format(args.D_learning_rate))
    logging.info('D_beta_1: {}'.format(args.D_beta_1))
    logging.info('D_beta_2: {}'.format(args.D_beta_2))

    ###################################
    ## GRABBING FILENAMES FOR IMAGES ##
    ###################################
    lenses = glob.glob(os.path.join(os.getcwd(), 'data/DLS/Lenses/*'))
    lenses = np.array(lenses)

    ##########################################################
    ## MAKING DATALOADERS FOR TRAINING, VALIDATION, TESTING ##
    ##########################################################
    train_dataloader = dataloader(
        filenames=lenses,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_parallel_calls=args.num_parallel_calls,
        buffer_size=args.buffer_size,
        prefetch=args.prefetch,
    )

    #######################################
    ## INITIALIZE MODEL, LOSSES, METRICS ##
    #######################################
    G = create_generator(args.z_dim, args.g_dim, 'Generator')
    D = create_discriminator(args.z_dim, args.d_dim, 'Discriminator')
    fixed_noise = tf.random.normal(shape=(16, args.z_dim))

    loss_object = tf.keras.losses.BinaryCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
    optimizerG = tf.keras.optimizers.Adam(
        learning_rate=args.G_learning_rate,
        beta_1=args.G_beta_1,
        beta_2=args.G_beta_2,
    )
    optimizerD = tf.keras.optimizers.Adam(
        learning_rate=args.D_learning_rate,
        beta_1=args.D_beta_1,
        beta_2=args.D_beta_2,
    )

    G.summary(print_fn=logging.info)
    D.summary(print_fn=logging.info)

    ##############################################
    ## Loss computations and training functions ##
    ##############################################
    @tf.function
    def discriminator_step(real_image):
        z = tf.random.normal(shape=(real_image.shape[0], args.z_dim))
        fake_image = G(z)

        with tf.GradientTape() as tape:
            epsilon = tf.random.uniform(
                shape=[fake_image.shape[0], 1, 1, 1], minval=0, maxval=1)
            interpolated_image = epsilon * \
                fake_image + (1-epsilon)*real_image
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_image)
                d_interpolated = D(interpolated_image)
            grad_d = gp_tape.gradient(
                d_interpolated, interpolated_image)
            slopes = tf.sqrt(
                1e-8 + tf.reduce_sum(tf.square(grad_d), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes-1.) ** 2)

            d_fake = D(fake_image)
            d_real = D(real_image)

            d_loss = tf.reduce_mean(
                d_fake) - tf.reduce_mean(d_real) + args.LAMBDA * gradient_penalty
        gradients = tape.gradient(d_loss, D.trainable_variables)
        optimizerD.apply_gradients(
            zip(gradients, D.trainable_variables))

        return d_loss

    @tf.function
    def generator_step():
        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(real_image.shape[0], args.z_dim))
            fake_image = G(z)
            d_fake = -D(fake_image)
            g_loss = tf.reduce_mean(d_fake)
            gradients = tape.gradient(g_loss, G.trainable_variables)
            optimizerG.apply_gradients(
                zip(gradients, G.trainable_variables))

            return g_loss

    ###################
    ## TRAINING LOOP ##
    ###################
    iter = 0
    d_loss = 0
    g_loss = 0
    for epoch in range(args.epochs):

        start = time.time()
        for real_image in train_dataloader:
            real_image = real_image[0]
            d_loss = discriminator_step(real_image)

            if iter % args.n_critic == 0:
                z = tf.random.normal(shape=(real_image.shape[0], args.z_dim))
                fake_image = G(z)
                g_loss = generator_step()
            # Save Models and Generated Images
            if iter % 100 == 0:
                G.save_weights(os.path.join(
                    model_path, 'G_iteration_{}.ckpt'.format(iter)))
                D.save_weights(os.path.join(
                    model_path, 'D_iteration_{}.ckpt'.format(iter)))

                images = G(fixed_noise)

                f, axarr = plt.subplots(4, 4, figsize=(8, 8), dpi=200)
                for i in range(16):
                    r = i // 4
                    c = i % 4
                    axarr[r, c].imshow(images[i])
                    axarr[r, c].axis('off')
                plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.05,
                                    right=0.95, bottom=0.05, top=0.95)
                plt.savefig(os.path.join(
                    image_path, 'iteration: {}.png'.format(iter)))
                plt.close()
            # Log run results
            template = 'Epoch {}, Iteration: {}, Discriminator Loss: {}, Generator Loss: {}, Elapsed Time: {}'
            logging.info(template.format(
                epoch,
                iter,
                d_loss,
                g_loss,
                time.time()-start)
            )
            iter += 1
            if iter == args.iterations:
                break
        if iter == args.iterations:
            break
        train_loss.reset_states()
