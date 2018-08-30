'''Main script file, loads/assembles features, GridSearch-like capability, creates/trains NN, loads NN for prediction. Also returns accuracy (which can be saved in Results dict),
 adding.'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)
print(SysPath)

from keras.models import load_model
from keras import backend as k
from result_recording.nn_results_saveNdisplay import cdnn_records_add
from utils.generators_multifiles import generator_train_bufferedseq_wfname, generator_test_bufferedseq_wfname
from utils.segmentanalysis import segmentphonelist, uttLD, segCorrect
from features_assembly.make_features import speakers_trainNtest
from features_assembly.select_features import select_trainNtest
from utils.review_phoneframesize import nseqsofnsize
from utils.scoreaccuracy import argmaxpredicts2forcedphones, argmaxpredicts2phones
from features_assembly.create_TrainTest_wordlvl_fromdbcorrect import createNcount_trainNtest
import numpy as np
import operator
from collections import defaultdict
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
k.set_session(tf.Session(config=config))



def main(testing=False):

    # Config Values, Feature Assembly, focused on WordCount/FrameCount
    Frame_length = 0.025
    Frame_step = 0.01

    overwrite_MFCCs = False
    TrainAll = False
    #testing = True
    FramelevelORword = False

    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    Dbdir = SysPath +'GOP-LSTM/PhoneInfo/speakers_db_correct/'

    '''Based on your computer setup. (I have both folders on same level as GOP-LSTM)'''
    Wavdir = SysPath + 'corpus/dat/speakers/'  # On SysPath level
    Holddir = SysPath + 'HoldDir/'

    Traindir = Holddir + 'Train/'
    Testdir = Holddir + 'Test/'
    PhoInfDir = SysPath + 'GOP-LSTM/PhoneInfo/'

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
            cdnn_dict_name = f'segmentation_records_{ByCount}.pk'
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
        cdnn_dict_name = f'segment_blstm_records_wl_{wordcount}.pk'
        print(f'Selected phones: {selected_phones}')
        print(f'Train count & test count: {totalcount}')

    if testing:
        cdnn_dict_name = f'testing_records.pk'
    print(f' Using:{cdnn_dict_name}')
    cdnn_address = SysPath + 'GOP-LSTM/Results/CDNN_segments/'
    # Iterate over gridsearch
    N_epochs = 70
    Input_tuple = (5, 26, 1)
    # ,[32 for _ in range(10)]
    ConvLayerList = [[32 for _ in range(10)]]
    DropoutList =[0.8]

    # add one for sil
    print(f'Number of phones:{N_classes}')
    N_classes += 1

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
                    print(f'No such model as :{Model_address}')

                # Forced Accuracy
                print('...Predicting...')
                if testing:
                    totaltest = 30
                gen = generator_test_bufferedseq_wfname(train_dir=Testdir,
                                                        batch_size=seq_size,
                                                        n_classes=N_classes,
                                                        wfname=True)


                ''' Need to add  dictionary with similar sounding word
                however it seems there are too few words to do this with a small test set.
                also, might be having problems with switching to 25 wc from 30wc'''
                this = True
                if this:
                    swddict = defaultdict(list)
                    for keys,v in w2pdict.items():
                        wordnum = keys.split('_')[-1]
                        if v not in swddict[wordnum]:
                            swddict[wordnum].append(v)

                    for keys,v in swddict.items():
                        v[0].append('_')
                        #print(keys,v)

                that = True
                if that:
                    perwordaccdict = defaultdict(list) # Word-key,list-accuracy, post-process
                    perwordmaxdict = defaultdict(list)  # Word-key,list-accuracy, post-process
                    skiplist = [17,18]
                    selected_phones.append('_')
                    diagnose = False
                    print(f'Total Test size:{totaltest}\n')
                    x, y, file = next(gen)
                    cfile = file
                    for _ in range(totaltest): # amount of words to be judged
                        if diagnose:
                            print(f'Word\'s phones{potphones}')
                        #print(f'Current file:{file}')
                        trueword = file.split('.')[0].split('_')[1]
                        #print(trueword)
                        segcount = 0
                        gwordphones = [] # gold standard word segments
                        # we need to do this for all words, not just expected
                        predictions = model.predict(x=x)
                        segcount += 1
                        gstd = argmaxpredicts2phones(y[0], selected_phones)
                        twordlistdict = defaultdict(list)
                        twordscoredict = defaultdict(list)
                        twordaccdict = defaultdict(list)
                        for index, plist in swddict.items():
                            twordlistdict[index] = []
                            twordscoredict[index] = 0
                            twordaccdict[index] = 0
                            if index not in skiplist:
                                for iindex, wsp in enumerate(plist): # should only be one, two later
                                    wspind = [selected_phones.index(sp) for sp in wsp]
                                    tv, tswd = argmaxpredicts2forcedphones(predictions[0], selected_phones, wspind,fwords=True)
                                    twordlistdict[index].append(tswd)
                                    twordscoredict[index] += tv

                        gwordphones += gstd
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
                            gstd = argmaxpredicts2phones(y[0], selected_phones)
                            for index, plist in swddict.items():
                                if index not in skiplist:
                                    for iindex, wsp in enumerate(plist):  # should only be one, two later
                                        wspind = [selected_phones.index(sp) for sp in wsp]
                                        tv, tswd = argmaxpredicts2forcedphones(predictions[0], selected_phones, wspind,
                                                                               fwords=True)
                                        twordlistdict[index] += (tswd)
                                        twordscoredict[index] += tv
                            gwordphones += gstd
                            x, y, file = next(gen)
                            predictions = model.predict(x=x)
                            if cfile != file: # break out of word while loop
                                samefile = False
                                cfile = file
                        gseg = segmentphonelist(gwordphones)
                        startsil = gseg[-1][1] #Index of Silence
                        g_len = len(gwordphones[:startsil])

                        for index, _ in swddict.items():
                            if index not in skiplist:
                                tacc = segCorrect(gwordphones[:startsil], twordlistdict[index][0][:startsil])/g_len
                                twordaccdict[index] = tacc
                        sortedacc = sorted(twordaccdict.items(),key=operator.itemgetter(1),reverse=True)
                        sortedmax = sorted(twordscoredict.items(),key=operator.itemgetter(1),reverse=True)
                        for index,tuple in enumerate(sortedacc):
                            if tuple[0] == trueword:
                                perwordaccdict[trueword].append(index)
                        for index,tuple in enumerate(sortedmax):
                            if tuple[0] == trueword:
                                perwordmaxdict[trueword].append(index)
                        #print(sortedmax)
                        #print(sortedacc)
                    # Number of correct, binary score, then a relative score, the greater the worse
                    averageaccuracy = 0
                    numberoftrials = 0
                    averagelengthaway = 0
                    rankingsacc = []
                    rankingsmax = []
                    for _, alist in perwordaccdict.items():
                        for score in alist:
                            numberoftrials += 1
                            averagelengthaway += score
                            if score == 0:
                                averageaccuracy += 1
                            rankingsacc.append(score)
                    print(f'Avg num of distance from 0, {averagelengthaway/numberoftrials} using max accuracy')
                    print(f'Avg accuracy {averageaccuracy/numberoftrials}')
                    averageaccuracy = 0
                    averagelengthaway = 0
                    for _, mlist in perwordmaxdict.items():
                        for score in mlist:
                            numberoftrials += 1
                            averagelengthaway += score
                            if score == 0:
                                averageaccuracy += 1
                            rankingsmax.append(score)
                    print(f'Avg num of distance from 0, {averagelengthaway/numberoftrials} using sum of max')
                    print(f'Avg sum phone maxs {averageaccuracy/numberoftrials}')
                    print(f'Out {len(perwordaccdict.keys())} words')

                    plt.subplot(1,2,1)
                    plt.title('Rankings by Accuracy')
                    plt.hist(rankingsacc,bins=38)
                    plt.xlabel('Occurrences')
                    plt.xlabel('Distance')
                    plt.subplot(1,2,2)
                    plt.title('Rankings by Sum of Softmax')
                    plt.hist(rankingsmax,bins=38)
                    plt.xlabel('Distance')
                    plt.tight_layout()
                    plt.show()



                del gen
                del model
                k.clear_session()



if __name__ == "__main__":
    main()
