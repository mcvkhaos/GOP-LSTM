# CNN BiLSTM
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)

from keras.models import Sequential, load_model
from keras import backend as k
from keras.layers import Dense, Flatten, TimeDistributed, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add
from utils.generators_multifiles import generator_train_bufferedseq_wfname, generator_test_bufferedseq_wfname
from utils.segmentanalysis import segmentphonelist, uttLD, segCorrect
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from utils.review_phoneframesize import nseqsofnsize
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
k.set_session(tf.Session(config=config))


def train_model(n_epochs, model, traindir, model_name, n_classes, totalsamples, dict_name, results_dir, batch_size=16,testing=False):
    print('...Training...')
    if testing:
        totalsamples = 10
        n_epochs = 3

    # due the variability of samples in each mfcc files batch size must be 1
    gen = generator_train_bufferedseq_wfname(train_dir=traindir,
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
        totalsamples = 10

    gen = generator_test_bufferedseq_wfname(train_dir=testdir,
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


def predict_model(model, testdir, n_classes, totalsamples, w2pdict, batch_size=16,testing=False):
    print('...Evaluating...')
    if testing:
        totalsamples = 10
    gen = generator_train_bufferedseq_wfname(train_dir=testdir,
                                             batch_size=batch_size,
                                             n_classes=n_classes)
    for s in range(totalsamples):
        x,y,file = next(gen)
        print(file)
        fname = file.split('.')[0]
        potphones = w2pdict[fname]
        print(potphones)
        predictions = model.predict_proba(x=x)
        print(predictions)
    return


def make_CNNLSTM_classifier(input_tuple, conv_layers, n_classes,seq_size, dropout_rate=0.0,loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], n_conv=3, channel_order='channels_last'):
    model = Sequential()
    print(f'Input tuple: {input_tuple}')
    new_input = (seq_size,input_tuple[0],input_tuple[1],input_tuple[2])
    print(new_input)
    model.add(TimeDistributed(Conv2D(filters=conv_layers[0],
                                     kernel_size=(n_conv, n_conv),
                                     strides=(1, 1),
                                     padding='same',
                                     data_format=channel_order,
                                     input_shape=input_tuple),
                                     input_shape=new_input))
    model.add(TimeDistributed(BatchNormalization()))

    for cl in conv_layers[1:]:
        model.add(TimeDistributed(Conv2D(filters=cl,
                                         kernel_size=(n_conv, n_conv),
                                         strides=(1, 1),
                                         padding='same',
                                         data_format=channel_order)))
        model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='valid')))
    model.add(TimeDistributed(Flatten()))
    flatten_nodes = int(model.layers[-1].output.shape[-1])
    model.add(Bidirectional(LSTM(units=flatten_nodes,
                                 activation='tanh',
                                 dropout=dropout_rate,
                                 return_sequences=True)))
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
    FramelevelORword = False

    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    Wavdir = SysPath + 'corpus/dat/speakers/'
    Dbdir = SysPath +'GOP-LSTM/PhoneInfo/speakers_db_correct/'
    Holddir = SysPath + 'HoldDir/'
    Traindir = Holddir + 'Train/'
    Testdir = Holddir + 'Test/'
    PhoInfDir = SysPath + 'GOP-LSTM/PhoneInfo/'

    N_context = 2
    N_ceps = 26
    wordcount = 5

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
                                                                       phoinfdir=PhoInfDir,
                                                                       wordcount=wordcount)
        N_classes = len(selected_phones)
        Traindir = Holddir + f'FLP_Train_{wordcount}/'
        Testdir = Holddir + f'FLP_Test_{wordcount}/'
        cdnn_dict_name = f'dcp_blstm_records_wl_{wordcount}.pk'
        print(f'Selected phones: {selected_phones}')
        print(f'Train count & test count: {totalcount}')

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    print(f' Using:{cdnn_dict_name}')
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'
    # Iterate over gridsearch
    N_epochs = 70
    Input_tuple = (5, 26, 1)
    # ,[32 for _ in range(10)]
    ConvLayerList = [[32 for _ in range(10)]]
    DropoutList =[0.8]

    # add one for sil
    N_classes += 1
    selected_phones.append('_')

    seq_sizelist = [64]
    for seq_size in seq_sizelist:
        totaltrain = nseqsofnsize(Traindir,seq_size=seq_size)
        totaltest = nseqsofnsize(Testdir,seq_size=seq_size)
        for cl in ConvLayerList:
            for dl in DropoutList:
                # Compile Params
                #cname = '_'.join(str(x) for x in cl)
                cname = f'{cl[0]}_x{len(cl)}'
                Model_name = f'BLSTM_CP{cname}_FBN_SS{seq_size}_DL{dl}_V2'
                # check if model exist with said name
                Model_address = f'{Model_name}model.hdf5'
                if os.path.exists(Model_address):
                    print(f'loading model: {Model_address}')
                    model = load_model(Model_address)
                    firsttime = False
                    print(f'Loaded')
                else:
                    model = make_CNNLSTM_classifier(input_tuple=Input_tuple,
                                                    conv_layers=cl,
                                                    n_classes=N_classes,
                                                    seq_size=seq_size,
                                                    dropout_rate=dl,
                                                    channel_order='channels_last')
                    model, Model_name2 = train_model(n_epochs=N_epochs,
                                                    model=model,
                                                    traindir=Traindir,
                                                    model_name=Model_name,
                                                    n_classes=N_classes,
                                                    totalsamples=totaltrain,
                                                    dict_name=cdnn_dict_name,
                                                    results_dir=cdnn_address,
                                                    batch_size=seq_size,
                                                    testing=testing)
                    print('...Evaluating...')
                    evaluate_model(model=model,
                                   testdir=Testdir,
                                   n_classes=N_classes,
                                   totalsamples=totaltest,
                                   model_name=Model_name2,
                                   dict_name=cdnn_dict_name,
                                   results_dir=cdnn_address,
                                   batch_size=seq_size,
                                   testing=testing)
                    firsttime = True
                if firsttime:
                    model.save(Model_address)

# Forced Accuracy
                print('...Predicting...')
                if testing:
                    totaltest = 30
                gen = generator_test_bufferedseq_wfname(train_dir=Testdir,
                                                        batch_size=seq_size,
                                                        n_classes=N_classes,
                                                        wfname=True)


                ''' Return Word Accuracy (by Softmax & ForcedMax), Max Seg Accuracy (from goldstandard-gst)'''
                that = True
                if that:
                    selected_phones.append('_')
                    diagnose = False
                    s_correct = 0
                    f_correct = 0
                    total = 0
                    s_IDS = 0
                    f_IDS = 0
                    maxsegtotal = 0
                    s_seg = 0
                    f_seg = 0
                    print(f'Total Test size:{totaltest}\n')
                    x, y, file = next(gen)
                    cfile = file
                    for _ in range(totaltest):  # amount of words to be judged
                        if diagnose:
                            print(f'Current file:{file}')
                            print(f'Word\'s phones{potphones}')
                        print(file)
                        segcount = 0
                        gwordphones = []  # gold standard word segments
                        swordphones = []  # softmax word segments
                        fwordphones = []  # forced word segments
                        fname = file.split('.')[0]
                        potphones = w2pdict[fname]
                        potphones.append('_')
                        pind = [selected_phones.index(sp) for sp in potphones]
                        predictions = model.predict(x=x)
                        segcount += 1
                        gstd = [selected_phones[sp] for sp in np.argmax(y, axis=2)[0]]
                        softmax = [selected_phones[sp] for sp in np.argmax(predictions, axis=2)[0]]
                        forceda = [selected_phones[pind[sp]] for sp in
                                   np.argmax(predictions[:, :, pind][0], axis=1)]
                        gwordphones += gstd
                        swordphones += softmax
                        fwordphones += forceda
                        x, y, file = next(gen)
                        predictions = model.predict(x=x)
                        if cfile == file:  # break out of word while loop
                            samefile = True
                            cfile = file
                        else:
                            samefile = False
                            cfile = file
                        while samefile:  # track error for same file
                            segcount += 1
                            gstd = [selected_phones[sp] for sp in np.argmax(y, axis=2)[0]]
                            softmax = [selected_phones[sp] for sp in np.argmax(predictions, axis=2)[0]]
                            forceda = [selected_phones[pind[sp]] for sp in
                                       np.argmax(predictions[:, :, pind][0], axis=1)]
                            gwordphones += gstd
                            swordphones += softmax
                            fwordphones += forceda
                            x, y, file = next(gen)
                            predictions = model.predict(x=x)
                            if cfile != file:  # break out of word while loop
                                samefile = False
                                cfile = file
                        # got word segs, process them
                        gseg = segmentphonelist(gwordphones)
                        sseg = segmentphonelist(swordphones)
                        fseg = segmentphonelist(fwordphones)
                        sLD = uttLD(gseg, sseg)
                        fLD = uttLD(gseg, fseg)
                        s_IDS += sLD
                        f_IDS += fLD
                        if diagnose:
                            print('\n')
                            print(gseg)
                            print(sseg)
                            print(fseg)
                            print('\n')
                            print(sLD)
                            print(fLD)
                            print('\n')
                        # accuracy
                        startsil = gseg[-1][1]  # Index of Silence
                        g_len = len(gwordphones[:startsil])
                        s_correct += segCorrect(gwordphones[:startsil], swordphones[:startsil])
                        f_correct += segCorrect(gwordphones[:startsil], fwordphones[:startsil])
                        total += g_len

                        # max-seg-score with known boundaries
                        # per word, then test set score
                        wordweight = 0
                        for seg in gseg[:-1]:  # last phone is silence '_'
                            maxsegtotal += 1
                            cphone = seg[0]
                            sboundedlist = swordphones[seg[1]:seg[2]]
                            fboundedlist = fwordphones[seg[1]:seg[2]]
                            smaxphone = max(sboundedlist, key=sboundedlist.count)
                            fmaxphone = max(fboundedlist, key=fboundedlist.count)
                            if smaxphone == cphone:
                                s_seg += 1
                            if fmaxphone == cphone:
                                f_seg += 1
                            if diagnose:
                                print(seg)
                                print(smaxphone, fmaxphone, cphone)

                    sLDpercent = sLD / total * 100
                    fLDpercent = fLD / total * 100
                    print(f'Insertions, Deletions, Substitions (SM):{sLD} out of {total}: {sLDpercent}%')
                    print(f'Insertions, Deletions, Substitions (FM):{fLD} out of {total}: {fLDpercent}%')
                    Spercent = s_correct / total * 100
                    Fpercent = f_correct / total * 100
                    print('\n')
                    print(f'Softmax: {s_correct} out of {total}, {Spercent}%')
                    print(f'Forced: {f_correct} out of {total}, {Fpercent}%')

                    Spercent = s_seg / maxsegtotal * 100
                    Fpercent = f_seg / maxsegtotal * 100

                    print(f'Softmax (seg): {s_seg} out of {maxsegtotal}, {Spercent}%')
                    print(f'Forced (seg): {f_seg} out of {maxsegtotal}, {Fpercent}%')
                    cdnn_records_add(loss=Spercent,
                                     accuracy=Fpercent,
                                     model_name=Model_name,
                                     nn_records_name=cdnn_dict_name,
                                     results_address=cdnn_address)
                del gen
                del model
                k.clear_session()



if __name__ == "__main__":
    main()
