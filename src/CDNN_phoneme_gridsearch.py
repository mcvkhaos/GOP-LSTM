import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys


SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)

from keras import  backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add, cdnn_records_rankNprint, cdnn_records_check
from utils.generators_multifiles import generator_train_batch
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
k.set_session(tf.Session(config=config))


def train_model(n_epochs, model, traindir, model_name, n_classes, totalsamples, dict_name, results_dir, batch_size=32,testing=False):
    print('...Training...')
    if testing:
        totalsamples = 10
        n_epochs = 3

    # due the variability of samples in each mfcc files batch size must be 1
    gen = generator_train_batch(train_dir=traindir,
                                batch_size=batch_size,
                                n_classes=n_classes)
    earlystop = EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=3,
                              verbose=1)
    history = model.fit_generator(generator=gen,
                                  steps_per_epoch=totalsamples,
                                  epochs=n_epochs,
                                  verbose=1,
                                  callbacks=[earlystop])
    loss = history.history['loss']
    acc = history.history['acc']
    print(f'Model fit history:{history.history}')
    print(f'Trained on {totalsamples} files for over {n_epochs} epochs.')
    print(f'Results directory: {results_dir}')
    model_name = model_name + f'_E{len(loss)}'
    cdnn_records_add(loss=loss,
                     accuracy=acc,
                     model_name=model_name,
                     nn_records_name=dict_name,
                     results_address=results_dir)
    del gen
    return model, model_name


def evaluate_model(model, testdir, n_classes, totalsamples, model_name, dict_name,results_dir, batch_size=32,testing=False):
    print('...Evaluating...')
    if testing:
        totalsamples = 100

    gen = generator_train_batch(train_dir=testdir,
                                batch_size=batch_size,
                                n_classes=n_classes)
    history = model.evaluate_generator(generator=gen,
                                       steps=totalsamples)
    loss,acc = history
    print(f'Model fit history:{history}')
    print(f'Trained on {totalsamples} files.')
    print(f'Results directory: {results_dir}')
    cdnn_records_add(loss=loss,
                     accuracy=acc,
                     model_name=model_name,
                     nn_records_name=dict_name,
                     results_address=results_dir)
    del gen
    return


def make_CDNN_classifier(input_tuple, conv_layers, dense_layers, n_classes, dropout_rate=0.1,loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], n_conv=3, channel_order='channels_last'):
    model = Sequential()
    model.add(Conv2D(filters=conv_layers[0],
                     activation='relu',
                     kernel_size=(n_conv, n_conv),
                     strides=(1, 1),
                     padding='same',
                     data_format=channel_order,
                     input_shape=input_tuple))
    # Convolutional Layers
    for cl in conv_layers[1:]:
        model.add(Conv2D(filters=cl,
                         activation='relu',
                         kernel_size=(n_conv, n_conv),
                         strides=(1, 1),
                         padding='same',
                         data_format=channel_order))
        model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Dense Layers
    for dl in dense_layers:
        model.add(Dense(int(dl[0])))
        if len(dl) > 1:
            model.add(Activation(dl[1]))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(int(n_classes),activation='softmax'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    print(model.summary())
    return model



def main(testing=False):

    # Config Values[DNN params]
    Frame_length = 0.025
    Frame_step = 0.01
    overwrite_MFCCs = False
    TrainAll = False
    #testing = True
    FramelevelORword = True


    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    Wavdir = SysPath + 'corpus/dat/speakers/'
    Dbdir = SysPath + 'GOP-LSTM/PhoneInfo/speakers_db_correct/'
    Holddir = SysPath + 'HoldDir/'
    Traindir = Holddir + 'Train/'
    Testdir = Holddir + 'Test/'
    N_context = 2
    N_ceps = 26
    wordcount = 30


    # Training & Test Data
    if FramelevelORword:
        speakers_trainNtest(db_corpus=Dbdir,
                            wav_corpus=Wavdir,
                            n_ceps=N_ceps,
                            n_context=N_context,
                            frame_length=Frame_length,
                            frame_step=Frame_step,
                            inmat=True,
                            holddir=Holddir,
                            overwrite=overwrite_MFCCs)
        cdnn_dict_name = 'cdnn_gridsearch_records_ALL.pk'
        if not TrainAll:
            ByCount = 8000
            Traindir = Holddir + 'Train_Correct/'
            Testdir = Holddir + 'Test_Correct/'
            selected_phones, totalcount = select_trainNtest(bycount=ByCount,
                                                            holddir=Holddir,
                                                            train_corpus=Traindir,
                                                            test_corpus=Testdir,
                                                            overwrite=False)
            Traindir = Holddir + f'Train_Select_{ByCount}/'
            Testdir = Holddir + f'Test_Select_{ByCount}/'
            N_classes = len(selected_phones)
            print(f'N selected classes: {N_classes}')
            cdnn_dict_name = f'cddnn_gridsearch_records_{ByCount}.pk'
    else:
        selected_phones,totalcount = createNcount_trainNtest(frame_length=Frame_length,
                                                             frame_step=Frame_step,
                                                             n_ceps=N_ceps,
                                                             n_context=N_context,
                                                             dbdir=Dbdir,
                                                             datdir=Wavdir,
                                                             holddir=Holddir,
                                                             wordcount=wordcount)
        N_classes = len(selected_phones)
        Traindir = Holddir + f'FLP_Train_{wordcount}/'
        Testdir = Holddir + f'FLP_Test_{wordcount}/'

        cdnn_dict_name = f'cddnn_gridsearch_records_wl_{wordcount}.pk'

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'

    print(f'total counts: {totalcount}')
    # Iterate over gridsearch
    N_epochs = 70
    Input_tuple = (5, 26, 1)
    Image_size = Input_tuple[0] * Input_tuple[1]
    ConvLayerList = [[64 for _ in range(10)],[32 for _ in range(10)]]
    #DenseLayerList = [[[1024, 'relu'], [512]],[[256,'relu'],[128]]]
    DenseLayerList = [[[1024]], [[512]]]
    DropoutList = [0.8]

    for cl in ConvLayerList:
        for do in DropoutList:
            for dl in DenseLayerList:
                tdl = dl.copy()
                tdl.insert(0, [cl[-1] * Image_size, 'relu'])
                # Model Creation or Loading
                if not os.path.isdir('Models'):
                    os.mkdir('Models')
                # Compile Params
                cname = '_'.join(str(x) for x in cl)
                dname = '_'.join(str(x) for X in dl for x in X)
                Model_name = f'CP{cname}_DND{dname}_DO{do}'

                # Check if model accounts
                if not cdnn_records_check(Model_name,cdnn_dict_name,cdnn_address):
                    print(f'Creating & training:  {Model_name}')
                    model = make_CDNN_classifier(input_tuple=Input_tuple,
                                                 conv_layers=cl,
                                                 dense_layers=tdl,
                                                 n_classes=N_classes,
                                                 dropout_rate=do,
                                                 channel_order='channels_first')
                    model, Model_name = train_model(n_epochs=N_epochs,
                                                    model=model,
                                                    traindir=Traindir,
                                                    model_name=Model_name,
                                                    n_classes=N_classes,
                                                    totalsamples=totalcount[0],
                                                    dict_name=cdnn_dict_name,
                                                    results_dir=cdnn_address,
                                                    testing=testing)
                    evaluate_model(model=model,
                                   testdir=Testdir,
                                   n_classes=N_classes,
                                   totalsamples=totalcount[1],
                                   model_name=Model_name,
                                   dict_name=cdnn_dict_name,
                                   results_dir=cdnn_address,
                                   testing=testing)
                    del model
                    k.clear_session()


                else:
                    print(f'{Model_name} has already been trained, skipping...')



    cdnn_records_rankNprint(nn_record_name=cdnn_dict_name,results_address=cdnn_address)




if __name__ == "__main__":
    main()
