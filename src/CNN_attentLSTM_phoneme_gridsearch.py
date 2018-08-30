import os
import sys
SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)
from keras.models import Sequential
from keras import backend as k
from keras.layers import Flatten, TimeDistributed, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add, cdnn_records_rankNprint
from utils.generators_multifiles import generator_train_bufferedseq, generator_train_bufferedseq_wfname
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from utils.review_phoneframesize import nseqsofnsize
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest

from utils.attention_decoder import AttentionDecoder
import numpy as np



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
    del gen
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
    del gen
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
    # multilayers of BiLSTM
    model.add(Bidirectional(LSTM(units=flatten_nodes,
                                 activation='tanh',
                                 dropout=dropout_rate,
                                 return_sequences=True)))
    model.add(AttentionDecoder(64,int(n_classes)))
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
    testing = True
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
    wordcount = 25

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
        selected_phones, totalcount, w2pdict = createNcount_trainNtest(frame_length=Frame_length,
                                                                       frame_step=Frame_step,
                                                                       n_ceps=N_ceps,
                                                                       n_context=N_context,
                                                                       dbdir=Dbdir,
                                                                       datdir=Wavdir,
                                                                       holddir=Holddir,
                                                                       wordcount=wordcount,
                                                                       overwrite=False)
        N_classes = len(selected_phones)
        Traindir = Holddir + f'FLP_Train_{wordcount}/'
        Testdir = Holddir + f'FLP_Test_{wordcount}/'
        print(f'Selected phones: {selected_phones}')
        print(f'Train count \& test count: {totalcount}')
        cdnn_dict_name = f'cp_attentlstm_gridsearch_records_wl_{wordcount}_V2.pk'

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'
    # Iterate over gridsearch
    N_epochs = 70
    Input_tuple = (5, 26, 1)
    ConvLayerList = [[8],[32]]
    DropoutList =[0.6,0.8]

    seq_sizelist = [32]
    for seq_size in seq_sizelist:
        totaltrain = nseqsofnsize(Traindir,seq_size=seq_size)
        totaltest = nseqsofnsize(Testdir,seq_size=seq_size)
        for cl in ConvLayerList:
            for dl in DropoutList:
                # Compile Params
                cname = '_'.join(str(x) for x in cl)
                Model_name = f'AttentBLSTM_CP{cname}_FBN_SS{seq_size}_DL{dl}'
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
                                                totalsamples=totaltrain,
                                                dict_name=cdnn_dict_name,
                                                results_dir=cdnn_address,
                                                batch_size=seq_size,
                                                testing=testing)
                evaluate_model(model=model,
                               testdir=Testdir,
                               n_classes=N_classes,
                               totalsamples=totaltest,
                               model_name=Model_name,
                               dict_name=cdnn_dict_name,
                               results_dir=cdnn_address,
                               batch_size=seq_size,
                               testing=testing)
                print('...Predicting...')
                if testing:
                    totaltest = 30
                gen = generator_train_bufferedseq_wfname(train_dir=Testdir,
                                                         batch_size=seq_size,
                                                         n_classes=N_classes)
                p_correct = 0
                f_correct = 0
                total = 0
                for s in range(totaltest):
                    x, y, file = next(gen)
                    fname = file.split('.')[0]
                    potphones = w2pdict[fname]
                    pind = [selected_phones.index(sp) for sp in potphones]
                    predictions = model.predict_proba(x=x)
                    TrueY = [selected_phones[sp] for sp in np.argmax(y, axis=2)[0]]
                    PredY = [selected_phones[sp] for sp in np.argmax(predictions, axis=2)[0]]
                    ForcY = [selected_phones[pind[sp]] for sp in np.argmax(predictions[:, :, pind][0], axis=1)]
                    p_correct += len([1 for x, y in zip(TrueY, PredY) if x == y])
                    f_correct += len([1 for x, y in zip(TrueY, ForcY) if x == y])
                    total += len(TrueY)
                p_percent = p_correct / total * 100
                f_percent = f_correct / total * 100
                cdnn_records_add(loss=p_percent,
                                 accuracy=f_percent,
                                 model_name=Model_name,
                                 nn_records_name=cdnn_dict_name,
                                 results_address=cdnn_address)
                print(f'Predicted correct:{p_correct} out of {total}, {p_percent}')
                print(f'Forced  correct:{f_correct} out of {total}, {f_percent}')

                diagnosis = False
                if diagnosis:
                    print(potphones)
                    print(file)
                    print(f'Goldstd:{[selected_phones[sp] for sp in np.argmax(y, axis=2)[0]]}')
                    print(f'Max All:{[selected_phones[sp] for sp in np.argmax(predictions,axis=2)[0]]}')
                    print(
                        f'ForcedA:{[selected_phones[pind[sp]] for sp in np.argmax(predictions[:, :, pind][0], axis=1)]}')
                del gen
                del model
                k.clear_session()


    cdnn_records_rankNprint(nn_record_name=cdnn_dict_name,results_address=cdnn_address)




if __name__ == "__main__":
    main()
