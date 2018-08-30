import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys

SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)

from keras import  backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add, cdnn_records_rankNprint, cdnn_records_check
from utils.generators_multifiles import generator_train_flatbatch
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
    gen = generator_train_flatbatch(train_dir=traindir,
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

    gen = generator_train_flatbatch(train_dir=testdir,
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


def make_CDNN_classifier(input_tuple, dense_layers, n_classes, dropout_rate=0.1,loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    model = Sequential()
    print(input_tuple)
    # Dense Layers
    first_nodes = input_tuple[0]*input_tuple[1]
    model.add(Dense(units=first_nodes,input_shape=(first_nodes,)))
    for dl in dense_layers:
        print(dl)
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
    Dbdir = './speakers_db_correct/'

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
        cdnn_dict_name = 'dnn_gridsearch_records_ALL.pk'
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
            cdnn_dict_name = f'dnn_gridsearch_records_{ByCount}.pk'
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

        cdnn_dict_name = f'dnn_gridsearch_records_wl{wordcount}.pk'

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'

    print(f'total counts: {totalcount}')
    # Iterate over gridsearch
    N_epochs = 50
    Input_tuple = (5, 26, 1)
    layers1 = [[2048,'relu'] for _ in range(0,6)]
    layers2 = [[2048,'relu'] for _ in range(0,4)]
    layers3 = [[2048,'relu'] for _ in range(0,2)]
    layers4 = [[1024,'relu'] for _ in range(0,4)]
    layers5 = [[512,'relu'] for _ in range(0,3)]
    DenseLayerList = [layers1,layers2,layers3,layers4,layers5]
    DropoutList = [0.2,0.8]

    for do in DropoutList:
        for dl in DenseLayerList:
            # Model Creation or Loading
            if not os.path.isdir('Models'):
                os.mkdir('Models')
            # Compile Params
            dname = '_'.join(str(x) for X in dl for x in X)
            Model_name = f'DND{dname}_DO{do}'

            # Check if model accounts
            if not cdnn_records_check(Model_name,cdnn_dict_name,cdnn_address):
                print(f'Creating & training:  {Model_name}')
                model = make_CDNN_classifier(input_tuple=Input_tuple,
                                             dense_layers=dl,
                                             n_classes=N_classes,
                                             dropout_rate=do)
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
