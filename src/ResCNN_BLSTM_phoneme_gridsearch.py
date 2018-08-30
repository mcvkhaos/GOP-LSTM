import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys

SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)

from keras.models import Model, load_model
from keras import backend as k
from keras.layers import add as kadd
from keras.layers import Dense, Flatten, TimeDistributed, LSTM, Bidirectional, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from result_recording.nn_results_saveNdisplay import cdnn_records_add
from utils.generators_multifiles import generator_train_bufferedseq_wfname, generator_test_bufferedseq_wfname
from utils.segmentanalysis import segmentphonelist,uttLD, segCorrect
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from utils.review_phoneframesize import nseqsofnsize
from utils.scoreaccuracy import argmaxpredicts2forcedphones,argmaxpredicts2phones
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest
import numpy as np
from collections import  defaultdict

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

    gen = generator_train_bufferedseq_wfname(train_dir=testdir,
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

def predict_model(model, testdir, n_classes, totalsamples, model_name, dict_name, results_dir, w2pdict, batch_size=16,testing=False):
    print('...Evaluating...')
    if testing:
        totalsamples = 10
    gen = generator_test_bufferedseq_wfname(train_dir=testdir,
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
    print(f'Input tuple: {input_tuple}')
    new_input = (seq_size,input_tuple[0],input_tuple[1],input_tuple[2])
    print(new_input)
    inputs = Input(shape=input_tuple)
    c = Conv2D(filters=conv_layers[0],
               kernel_size=(n_conv, n_conv),
               strides=(1, 1),
               padding='same',
               data_format=channel_order)(inputs)
    b = BatchNormalization()(c)
    added = kadd([inputs,b])
    for _ in range(0,conv_layers[1]):
        c = Conv2D(filters=conv_layers[0],
                   kernel_size=(n_conv, n_conv),
                   strides=(1, 1),
                   padding='same',
                   data_format=channel_order)(added)
        b = BatchNormalization()(c)
        added = kadd([added, b])
    mp = MaxPooling2D((2,2))(added)
    flatout = Flatten()(mp)
    Amodel = Model(inputs=inputs,outputs=flatout)
    all_input = Input(shape=new_input)
    encoded_audio_seq = TimeDistributed(Amodel)(all_input)
    encoded = Bidirectional(LSTM(units=conv_layers[2],
                                 activation='tanh',
                                 dropout=dropout_rate,
                                 return_sequences=True))(encoded_audio_seq)
    output = Dense(int(n_classes),activation='softmax')(encoded)
    Omodel = Model(inputs=all_input,outputs=output)
    Omodel.compile(loss=loss,
                   optimizer=optimizer,
                   metrics=metrics)
    print(Amodel.summary())
    print(Omodel.summary())
    return Omodel



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
    PhoInfDir = SysPath + 'GOP-LSTM/PhoneInfo/'
    N_context = 2
    N_ceps = 26
    wordcount = 10

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
        cdnn_dict_name = f'drescp_blstm_records_wl_{wordcount}.pk'
        print(f'Selected phones (amount: {selected_phones}')
        print(f'Train count & test count: {totalcount}')



    if testing:
        cdnn_dict_name = f'testing_records.pk'
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_phones/'
    # Iterate over gridsearch
    N_epochs = 80
    Input_tuple = (5, 26, 1)
    ConvLayerList = [[32,7,1024]]
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
                cname = '_'.join(str(x) for x in cl)
                Model_name = f'ResBLSTM_C{cname}_FBN_SS{seq_size}_DL{dl}_V3'
                Model_address = f'{Model_name}model.hdf5'
                if os.path.exists(Model_address):
                    print(f'Loading Model:{Model_address}')
                    model = load_model(Model_address)
                    firsttime = False
                    print('Loaded')
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


                that = True

                # possibly create function to address, which words are possibly useful forced-word-max
                #sameworddiffname
                swdname = ['9', '44']
                swdforms = [[['b','ɔ','l','_'],['b','a','l','_'],['b','ɔ','_']],[['d','ɹ','ʌ','m','_'],['dʒ','ɹ','ʌ','m','_']]]

                # import the whole list for testing each possible word, instead of softmax
                #totaltest = 3 # for testing purpose shorten
                if that:
                    perphoneaccdict = defaultdict(list)
                    selected_phones.append('_')
                    diagnose = False
                    s_correct = 0
                    f_correct = 0
                    fw_correct = 0
                    total = 0
                    s_IDS = 0
                    f_IDS = 0
                    maxsegtotal = 0
                    s_seg = 0
                    f_seg = 0

                    print(f'Total Test size:{totaltest}\n')
                    x, y, file = next(gen)
                    cfile = file
                    for _ in range(totaltest): # amount of words to be judged
                        if diagnose:
                            print(f'Current file:{file}')
                            print(f'Word\'s phones{potphones}')
                        segcount = 0
                        gwordphones = [] # gold standard word segments
                        swordphones = [] # softmax word segments
                        fwordphones = [] # forced word segments
                        fname = file.split('.')[0]
                        potphones = w2pdict[fname]
                        fnamelast = fname.split('_')[-1]
                        potphones.append('_')
                        pind = [selected_phones.index(sp) for sp in potphones]
                        predictions = model.predict(x=x)
                        segcount += 1
                        gstd = argmaxpredicts2phones(y[0], selected_phones)
                        softmax = argmaxpredicts2phones(predictions[0], selected_phones)
                        forceda = argmaxpredicts2forcedphones(predictions[0], selected_phones, pind)
                        wordmaxforce = []
                        if fnamelast in swdname:
                            wordmaxforce = swdforms[swdname.index(fnamelast)]
                            swdlist = []
                            swdlistscore = [0 for _ in wordmaxforce]
                            for index, wsp in enumerate(wordmaxforce):
                                wspind = [selected_phones.index(sp) for sp in wsp]
                                tv, tswd = argmaxpredicts2forcedphones(predictions[0], selected_phones, wspind,fwords=True)
                                swdlist.append(tswd)
                                swdlistscore[index] += tv

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
                            gstd = argmaxpredicts2phones(y[0],selected_phones)
                            softmax =  argmaxpredicts2phones(predictions[0],selected_phones)
                            forceda = argmaxpredicts2forcedphones(predictions[0],selected_phones,pind)
                            if wordmaxforce:
                                for index, wsp in enumerate(wordmaxforce):
                                    wspind = [selected_phones.index(sp) for sp in wsp]
                                    tv, tswd = argmaxpredicts2forcedphones(predictions[0], selected_phones, wspind,fwords=True)
                                    swdlist[index] += tswd
                                    swdlistscore[index] += tv
                            gwordphones += gstd
                            swordphones += softmax
                            fwordphones += forceda
                            x, y, file = next(gen)
                            predictions = model.predict(x=x)
                            if cfile != file: # break out of word while loop
                                samefile = False
                                cfile = file
                        # got word segs, process them

                        gseg = segmentphonelist(gwordphones)
                        sseg = segmentphonelist(swordphones)
                        fseg = segmentphonelist(fwordphones)
                        sLD = uttLD(gseg,sseg)
                        fLD = uttLD(gseg,fseg)
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
                        startsil = gseg[-1][1] #Index of Silence
                        g_len = len(gwordphones[:startsil])
                        s_correct += segCorrect(gwordphones[:startsil],swordphones[:startsil])
                        f_correct += segCorrect(gwordphones[:startsil],fwordphones[:startsil])
                        total += g_len
                        if wordmaxforce:
                            print(f'actual:{potphones}')
                            print(f'swd {wordmaxforce}') # list for individual comparision
                            print(f'scores{swdlistscore}')
                            for wlist in swdlist:
                                print(segCorrect(gwordphones[:startsil],wlist[:startsil])/g_len)
                            print('\n')


                        # max-seg-score with known boundaries
                        # per word, then test set score
                        wordweight = 0
                        for seg in gseg[:-1]: # last phone is silence '_'
                            maxsegtotal += 1
                            cphone = seg[0]
                            sboundedlist = swordphones[seg[1]:seg[2]]
                            fboundedlist = fwordphones[seg[1]:seg[2]]
                            smaxphone = max(sboundedlist, key=sboundedlist.count)
                            fmaxphone = max(fboundedlist,key=fboundedlist.count)
                            if smaxphone == cphone:
                                s_seg += 1
                            if fmaxphone == cphone:
                                f_seg += 1
                            #if cphone in perphoneaccdict.keys():
                            #    perphoneaccdict[cphone].append()
                            if diagnose:
                                print(seg)
                                print(smaxphone,fmaxphone, cphone)

                    sLDpercent = sLD/total*100
                    fLDpercent = fLD/total*100
                    print(f'Insertions, Deletions, Substitions (SM):{sLD} out of {total}: {sLDpercent}%')
                    print(f'Insertions, Deletions, Substitions (FM):{fLD} out of {total}: {fLDpercent}%')
                    Spercent = s_correct/total*100
                    Fpercent = f_correct/total*100
                    print('\n')
                    print(f'Softmax: {s_correct} out of {total}, {Spercent}%')
                    print(f'Forced: {f_correct} out of {total}, {Fpercent}%')

                    Spercent = s_seg / maxsegtotal * 100
                    Fpercent = f_seg / maxsegtotal * 100

                    print(f'Softmax (seg): {s_seg} out of {maxsegtotal}, {Spercent}%')
                    print(f'Forced (seg): {f_seg} out of {maxsegtotal}, {Fpercent}%')


                del gen
                del model
                k.clear_session()


if __name__ == "__main__":
    main()
