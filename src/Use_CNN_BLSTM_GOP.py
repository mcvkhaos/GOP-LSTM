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
import random

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
                Model_address = 'BLSTM_CP32_x5_FBN_SS64_DL0.8_V2model.hdf5'
                if os.path.exists(Model_address):
                    print(f'loading model: {Model_address}')

                    model = load_model(Model_address)
                    print(model.inputs,model.outputs)
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
                    sameworddiffprondict.pop('17')
                    sameworddiffprondict.pop('18')
                    for key,v in sameworddiffprondict.items():
                        v[0] +='_'
                        print(key,v)

                similardict = {'13':'48','48':'13','25':'5','5':'25'}
                dolist = ['13','48','25','5']

                ''' Return Word Accuracy (by Softmax & ForcedMax), Max Seg Accuracy (from goldstandard-gst)'''
                that = True
                # import the whole list for testing each possible word, instead of softmax
                totaltest = 15 # for testing purpose shorten
                if that:
                    selected_phones.append('_')
                    print(f'Total Test size:{totaltest}\n')
                    x, y, file = next(gen)
                    cfile = file
                    cfname = cfile.split('.')[0].split('_')[-1]
                    n = 0
                    for _ in range(totaltest): # amount of words to be judged
                        # Two versions, Right Word and Random Wrong Word
                        print(f'starting file:{file}, {cfname}')
                        cfile = file
                        cfname = cfile.split('.')[0].split('_')[-1]
                        if cfname not in dolist:
                            notinlist = True
                            while notinlist:
                                n += 1
                                x, y, file = next(gen)
                                cfile = file
                                cfname = cfile.split('.')[0].split('_')[-1]
                                if cfname in dolist:
                                    print(f'in')
                                    notinlist = False
                                    print(f'N:{n}')
                                elif n > 100:
                                    notinlist = False
                                    print(f'N:{n}')

                        predictions = []
                        samefile = True
                        while samefile:
                            tp = model.predict(x=x)[0].tolist()
                            predictions += tp
                            # get all values then roll through loops
                            x, y, file = next(gen)
                            if cfile != file:  # break out of word while loop
                                samefile = False
                                print(f'Next file:{file}')
                        print(f'Going in: {cfile},{cfname}')
                        for turnN in range(2):
                            sscores = []
                            fscores = []
                            fwordphones = []
                            if turnN == 1:
                                fname = similardict[cfname]
                                print(f'Wrong: {fname}')
                                potphones = sameworddiffprondict[fname][0]
                            else:
                                print(f'Right: {cfname}')
                                potphones = sameworddiffprondict[cfname][0]
                            print(potphones)
                            pind = [selected_phones.index(sp) for sp in potphones]
                            print(f'Expected word: {potphones}')
                            predictions = np.asarray(predictions)
                            smargs = np.max(predictions,axis=1)
                            fargs = np.max(predictions[:,pind],axis=1)
                            fwordphones += [selected_phones[pind[sp]] for sp in np.argmax(predictions[:,pind],axis=1)]
                            sscores += smargs.tolist()
                            fscores += fargs.tolist()
                            # got word segs, process them
                            fseg = segmentphonelist(fwordphones)

                            # use fseg to separate phones and then show their prob
                            fscores = np.asarray(fscores)
                            sscores = np.asarray(sscores)
                            gop_per_frame = np.round(np.abs(np.log(fscores/sscores)),4)
                            gop_list = []
                            avg_phone_error = []
                            for seg in fseg[:-1]: #skip space
                                phone_chuck = gop_per_frame[seg[1]-1:seg[2]-1]
                                phonescore = np.round(np.sum(phone_chuck)/seg[3],3)
                                gop_list.append((seg[0],phonescore,seg[3]))
                                avg_phone_error.append(phonescore)
                            print(gop_list)
                            print(np.round(np.average(avg_phone_error)))
                            print('\n')
                            # accuracy


                    # GOP, create ratios between RNN-WF/RNN-O and RNN-WF/(0.5*WF + 0.5*O)
                    # for correct words, for similar words, then for opposite words


                del gen
                del model
                k.clear_session()



if __name__ == "__main__":
    main()
