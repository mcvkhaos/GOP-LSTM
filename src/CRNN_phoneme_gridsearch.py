import os
import sys
SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add, cdnn_records_rankNprint
from utils.generators_multifiles import generator_train_bufferedseq
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from utils.review_phoneframesize import nseqsofnsize
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest



def train_model(n_epochs, model, traindir, model_name, n_classes, totalsamples, dict_name, results_dir, batch_size=16,testing=False):
    print('...Training...')
    if testing:
        totalsamples = 1
        n_epochs = 3

    # due the variability of samples in each mfcc files batch size must be 1
    gen = generator_train_bufferedseq(train_dir=traindir,
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
    return model, model_name


def evaluate_model(model, testdir, n_classes, totalsamples, model_name, dict_name,results_dir, batch_size=16,testing=False):
    print('...Evaluating...')
    if testing:
        totalsamples = 100

    gen = generator_train_bufferedseq(train_dir=testdir,
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
    return


def make_CNNLSTM_classifier(input_tuple, conv_layers, n_classes,seq_size, dropout_rate=0.0,loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], n_conv=3, channel_order='channels_last'):
    model = Sequential()
    print(f'Input tuple: {input_tuple}')
    new_input = (seq_size,input_tuple[0],input_tuple[1],input_tuple[2])
    print(new_input)
    for cl in conv_layers:
        model.add(TimeDistributed(Conv2D(filters=cl,
                                         kernel_size=(n_conv, n_conv),
                                         strides=(1, 1),
                                         padding='same',
                                         data_format=channel_order,
                                         input_shape=input_tuple),
                                  input_shape=new_input))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(BatchNormalization()))
    flatten_nodes = int(model.layers[-1].output.shape[-1])
    model.add(LSTM(units=flatten_nodes,
                   activation='tanh',
                   dropout=dropout_rate,
                   return_sequences=True))
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
    FramelevelORword = False

    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    Wavdir = SysPath + 'corpus/dat/speakers/'
    Dbdir = SysPath + 'GOP-LSTM/PhoneInfo/speakers_db_correct/'
    Holddir = SysPath + 'HoldDir/'
    Traindir = Holddir + 'Train/'
    Testdir = Holddir + 'Test/'
    N_context = 2
    N_ceps = 26
    wordcount = 40

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
        cdnn_dict_name = 'crnn_gridsearch_records_ALL.pk'
        if not TrainAll:
            ByCount = 4000
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
            cdnn_dict_name = f'crnn_gridsearch_records_{ByCount}.pk'
    else:
        selected_phones, totalcount = createNcount_trainNtest(frame_length=Frame_length,
                                                              frame_step=Frame_step,
                                                              n_ceps=N_ceps,
                                                              n_context=N_context,
                                                              dbdir=Dbdir,
                                                              datdir=Wavdir,
                                                              holddir=Holddir,
                                                              wordcount=wordcount,
                                                              overwrite=True)
        N_classes = len(selected_phones)
        Traindir = Holddir + f'FLP_Train_{wordcount}/'
        Testdir = Holddir + f'FLP_Test_{wordcount}/'

        cdnn_dict_name = f'crnn_gridsearch_records_wl_{wordcount}.pk'

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'
    print(totalcount)
    # Iterate over gridsearch
    N_epochs = 70
    Input_tuple = (5, 26, 1)
    ConvLayerList = [[8],[32]]
    DropoutList =[0.8,0.9]

    seq_sizelist = [64,128]
    for seq_size in seq_sizelist:
        totalcount = nseqsofnsize(Traindir,seq_size=seq_size)
        for cl in ConvLayerList:
            for dl in DropoutList:
                # Compile Params
                cname = '_'.join(str(x) for x in cl)
                Model_name = f'CP{cname}_FBN_SS{seq_size}_DL{dl}'
                model = make_CNNLSTM_classifier(input_tuple=Input_tuple,
                                                conv_layers=cl,
                                                n_classes=N_classes,
                                                seq_size=seq_size,
                                                dropout_rate=dl,
                                                channel_order='channels_last')
                model, Model_name = train_model(n_epochs=N_epochs,
                                                model=model,
                                                traindir=Traindir,
                                                model_name=Model_name,
                                                n_classes=N_classes,
                                                totalsamples=totalcount,
                                                dict_name=cdnn_dict_name,
                                                results_dir=cdnn_address,
                                                batch_size=seq_size,
                                                testing=testing)
                evaluate_model(model=model,
                               testdir=Testdir,
                               n_classes=N_classes,
                               totalsamples=totalcount,
                               model_name=Model_name,
                               dict_name=cdnn_dict_name,
                               results_dir=cdnn_address,
                               batch_size=seq_size,
                               testing=testing)
                del model

    cdnn_records_rankNprint(nn_record_name=cdnn_dict_name,results_address=cdnn_address)




if __name__ == "__main__":
    main()
