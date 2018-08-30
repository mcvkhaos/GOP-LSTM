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
from collections import defaultdict

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
    ConvLayerList = [[32 for _ in range(5)]]
    DropoutList =[0.8]

    # add one for sil
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
                # if manual entry
                Model_address = 'ResBLSTM_C32_7_1024_FBN_SS64_DL0.8_V3model.hdf5'
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
                    sameworddiffprondict = defaultdict(list)
                    for keys,v in w2pdict.items():
                        wordnum = keys.split('_')[-1]
                        #check if already in dict list
                        if v not in sameworddiffprondict[wordnum]:
                            sameworddiffprondict[wordnum].append(v)



                ''' Return Word Accuracy (by Softmax & ForcedMax), Max Seg Accuracy (from goldstandard-gst)'''
                that = True

                # possibly create function to address, which words are possibly useful forced-word-max
                #sameworddiffname
                swdname = ['9', '44']
                swdforms = [[['b','ɔ','l','_'],['b','a','l','_'],['b','ɔ','_']],[['d','ɹ','ʌ','m','_'],['dʒ','ɹ','ʌ','m','_']]]

                # import the whole list for testing each possible word, instead of softmax
                #totaltest = 3 # for testing purpose shorten
                if that:
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
                    wordset = set()
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
                        wordset.add(fnamelast)
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

                    print(f'Number of train: {totaltrain}')
                    print(f'Number of test: {totaltest}')
                    print(f'Number of phones:{N_classes-1}')
                    print(f'Number of words:{len(sameworddiffprondict.keys())}')
                    print(wordset)
                    print(len(wordset))


                del gen
                del model
                k.clear_session()



if __name__ == "__main__":
    main()
