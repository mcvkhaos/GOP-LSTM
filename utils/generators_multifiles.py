import os
import numpy as np
from keras.utils import to_categorical

# generates a simple sample at a time
def generator_train_single(train_dir):
    files = os.listdir(train_dir)
    while True:
        np.random.shuffle(files)
        for file in files:
            npzfile = np.load(train_dir + file)
            x_train = npzfile['indata']
            y_train = npzfile['labels']
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            order = np.arange(len(x_train))
            np.random.shuffle(order)
            for n in range(len(order)):
                yield x_train[n],y_train[n]

# batch generator
def generator_train_batch_xonly(train_dir,batch_size):
    g = generator_train_single(train_dir=train_dir)
    while True:
        batch = []
        for i in range(batch_size):
            x,_ = next(g)
            batch.append(x)
        batch = np.asarray(batch)
        yield batch,batch


def generator_train_batch(train_dir,batch_size,n_classes):
    g = generator_train_single(train_dir=train_dir)
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x,y = next(g)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = to_categorical(np.asarray(batch_y),num_classes=n_classes)
        yield batch_x,batch_y

def generator_train_flatbatch(train_dir,batch_size,n_classes):
    g = generator_train_single(train_dir=train_dir)
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x,y = next(g)
            batch_x.append(x.flatten())
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = to_categorical(np.asarray(batch_y),num_classes=n_classes)
        yield batch_x,batch_y

def generator_train_expandedbatch(train_dir,batch_size,n_classes):
    g = generator_train_single(train_dir=train_dir)
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x,y = next(g)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = to_categorical(np.asarray(batch_y),num_classes=n_classes)
        batch_x = np.expand_dims(batch_x,axis=0)
        batch_y = np.expand_dims(batch_y,axis=0)
        yield batch_x,batch_y


def generator_train_bufferedseq_wfname(train_dir,batch_size,n_classes,wfname=False):
    sil = n_classes-1
    files = os.listdir(train_dir)
    while True:
        np.random.shuffle(files)
        for file in files:
            npzfile = np.load(train_dir + file)
            x_train = npzfile['indata']
            y_train = npzfile['labels']
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            n_samples = len(y_train)
            n_batches = int(np.floor(n_samples/batch_size))
            if n_batches > 0:
                for n in range(n_batches):
                    x = x_train[n*batch_size:(n+1)*batch_size]
                    y = y_train[n*batch_size:(n+1)*batch_size]
                    y = to_categorical(y, n_classes)
                    x = np.expand_dims(x,axis=0)
                    y = np.expand_dims(y,axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y
                x = np.zeros((batch_size,x_train.shape[1],
                              x_train.shape[2],x_train.shape[3]))
                y = np.ones(batch_size)*sil
                left = n_samples - (batch_size * (n_batches))
                n += 1
                if left > 0:
                    x[:left] = x_train[n * batch_size:]
                    y[:left] = y_train[n * batch_size:]
                    y = to_categorical(y, n_classes)
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y
            else:
                x = np.zeros((batch_size, x_train.shape[1],
                              x_train.shape[2], x_train.shape[3]))
                y = np.ones(batch_size) * sil
                left = n_samples
                if left > 0:
                    x[:left] = x_train
                    y[:left] = y_train
                    y = to_categorical(y, num_classes=n_classes)
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y


def generator_test_bufferedseq_wfname(train_dir,batch_size,n_classes,wfname=False):
    sil = n_classes-1
    files = os.listdir(train_dir)
    while True:
        for file in files:
            npzfile = np.load(train_dir + file)
            x_train = npzfile['indata']
            y_train = npzfile['labels']
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            n_samples = len(y_train)
            n_batches = int(np.floor(n_samples/batch_size))
            if n_batches > 0:
                for n in range(n_batches):
                    x = x_train[n*batch_size:(n+1)*batch_size]
                    y = y_train[n*batch_size:(n+1)*batch_size]
                    y = to_categorical(y, n_classes)
                    x = np.expand_dims(x,axis=0)
                    y = np.expand_dims(y,axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y
                x = np.zeros((batch_size,x_train.shape[1],
                              x_train.shape[2],x_train.shape[3]))
                y = np.ones(batch_size)*sil
                left = n_samples - (batch_size * (n_batches))
                n += 1
                if left > 0:
                    x[:left] = x_train[n * batch_size:]
                    y[:left] = y_train[n * batch_size:]
                    y = to_categorical(y, n_classes)
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y
            else:
                x = np.zeros((batch_size, x_train.shape[1],
                              x_train.shape[2], x_train.shape[3]))
                y = np.ones(batch_size) * sil
                left = n_samples
                if left > 0:
                    x[:left] = x_train
                    y[:left] = y_train
                    y = to_categorical(y, num_classes=n_classes)
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                    if wfname:
                        yield x,y,file
                    else:
                        yield x,y


def generator_train_bufferedbatch(train_dir,batch_size,n_classes):
    g = generator_train_single(train_dir=train_dir)
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x,y = next(g)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = to_categorical(np.asarray(batch_y),num_classes=n_classes)
        batch_x = np.expand_dims(batch_x, axis=0)
        batch_y = np.expand_dims(batch_y, axis=0)
        yield batch_x,batch_y