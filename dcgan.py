from keras.models import Sequential
from keras.layers import Reshape, AveragePooling2D, Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, 5, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, 5, 5, padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, 5, 5, padding='same',\
                      input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*image_dim[0]:(i+1)* image_dim[0], j*image_dim[1]:\
        (j+1)*image_dim[1]] = image_batches[index, :, :].reshape(image_dim[0],\
         image_dim[1])]
    return image


def train(BATCH_SIZE):
    EPOCHS = 100
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype('float32')
    Y_train = y_train.reshape(y_train.shape[0], -1).astype('float32')

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    generator.compile(loss='binary_crossentropy', optimizer="ADAM")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer="SGD")
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer="ADAM")
    noise = np.zeros((BATCH_SIZE, 100))

    for epoch in range(EPOCHS):
        print("Epoch is", epoch)
        for offset in range(0,X_train.shape[0], BatchSize):
                end = offset + BatchSize
                batch_X_train, batch_Y_train = X_train[offset:end],\
                 Y_train[offset:end]
                #Gaussian noise between (-1, 1)
                mu, sigma = 0, 1
                lower, upper = -1, 1
                noise_batch = np.asarray([stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,\
                                           loc=mu,scale=sigma,size=LatentVectorSize)\
                                    for i in range(BatchSize)])
                GeneratedImages =  np.asarray(generator.predict(noise_batch,\
                 verbose=1))
                if offset%50==0 and Epoch%5==0:
                    SpriteImg =  combine_images(GeneratedImages)
                    Image.fromarray(SpriteImg, mode='RGB').save( \
                    "TrainingImages/Epoch_"+str(Epoch)+"_"+\
                                                   "Batch_"+str(offset)+".png")
                combined_X_train =  np.vstack((batch_X_train, GeneratedImages))
                combined_Y_train = np.hstack((np.random.uniform(0.7, 1.2, BatchSize), \
                                              np.asarray([0.0]*BatchSize)))
                d_loss = discriminator.train_on_batch(combined_X_train, combined_Y_train)
                discriminator.trainable = False
                g_loss = GAN.train_on_batch(noise_batch, np.ones(BatchSize))
                discriminator.trainable = True
                if  Epoch%2 == 0:
                    generator.save('Generator.h5')
                    discriminator.save('Discriminator.h5')

def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
