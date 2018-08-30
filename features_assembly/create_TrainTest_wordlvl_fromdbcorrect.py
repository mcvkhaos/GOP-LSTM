import os
import pickle as pk
from collections import OrderedDict, defaultdict
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
from features_assembly.make_features import wav2mfcc2inmat, db2labels
from utils.oddsNends import ifnodirmkdir_elsewarn

""" Word-level Train & Test creation"""
def createNcount_trainNtest(frame_length,frame_step,n_ceps,n_context, dbdir, datdir, wordcount,phoinfdir, holddir='',overwrite=False):
    mfa_train_dir = f'{holddir}MFA_Train_{wordcount}/'
    mfa_test_dir = f'{holddir}MFA_Test_{wordcount}/'
    flp_train_dir = f'{holddir}FLP_Train_{wordcount}/'
    flp_test_dir = f'{holddir}FLP_Test_{wordcount}/'
    """save selected phone and word2phonesdict
    locally to perserve across systems"""
    selected_phones_name = f'{phoinfdir}selected_phones_wl{wordcount}.pk'
    word2phonesdict_name = f'{phoinfdir}wordphones2dict_wl{wordcount}.pk'
    traintestlists_name = f'{phoinfdir}traintestlists_wl{wordcount}.pk'
    # Check if train folder exist, o.w. redo
    if overwrite or not os.path.exists(flp_train_dir):
        print('Creating Train_Select & Test_Select sets...')
        ifnodirmkdir_elsewarn(flp_train_dir)
        ifnodirmkdir_elsewarn(flp_test_dir)
        ifnodirmkdir_elsewarn(mfa_test_dir)
        ifnodirmkdir_elsewarn(mfa_train_dir)

        selectwordlist,wordsdict = surveydb2contdict(dbdir,wordcount,True)
        selectwordlist = sorted(selectwordlist)
        word2phonesdict = {}
        new_dictfile = mfa_train_dir + 'words.dict'
        with open(new_dictfile,'w', encoding='utf-8') as wf:
            for n in selectwordlist:
                wf.write(f'{n} ')
                wf.write(' '.join(x for x in n.split('_')))
                wf.write('\n')
        phonelist = list({x for word in selectwordlist for x in word.split('_')})
        phonelist = sorted(phonelist)
        print(phonelist)
        phonelist.append('.pau')
        phonelist.append('.exm')
        allspeakers = sorted(os.listdir(datdir))
        # save lists and check if these have been found before
        if os.path.exists(traintestlists_name):
            print(f' loading: {traintestlists_name} for Train/Testdirs')
            with open(traintestlists_name, 'rb') as rp:
                tdirs = pk.load(rp)
                traindirs = tdirs[0]
                testdirs = tdirs[1]
        else:
            print(f'  creating: {traintestlists_name}, first time')
            traindirs,testdirs = train_test_split(allspeakers,test_size=0.1,random_state=7)
            with open(traintestlists_name, 'wb') as wp:
                pk.dump([traindirs,testdirs], wp)
        print(f'Train:{traindirs} \n Test:{testdirs}')
        framecount = {}
        traincount = 0
        testcount = 0
        traindirs = sorted(traindirs)
        testdirs = sorted(testdirs)
        for word, addresslist in wordsdict.items():
            for spkutter in sorted(addresslist):
                spkr, utter = spkutter.split('_')
                if spkr in traindirs:
                    dest_mfa = mfa_train_dir + spkr + '/'
                    dest_flp = flp_train_dir
                    trainOtest = True
                elif spkr in testdirs:
                    dest_mfa = mfa_test_dir + spkr + '/'
                    dest_flp = flp_test_dir
                    trainOtest = False
                else:
                    continue
                if not os.path.isdir(dest_mfa):
                    os.mkdir(dest_mfa)
                dbfile = f'{dbdir}{spkr}/{utter}'
                utter = utter.split('.db')[0]
                labfile =f'{dest_mfa}{spkr}_{utter}.lab'
                with open(labfile,'w',encoding='utf-8') as wf:
                    wf.write(word)
                original_wav = datdir + spkr + '/' + utter + '.wav'
                dest_wav = f'{dest_mfa}{spkr}_{utter}.wav'
                copyfile(original_wav, dest_wav)
                indata, n_frames = wav2mfcc2inmat(wavfile=original_wav,
                                                  n_ceps=n_ceps,
                                                  n_context=n_context,
                                                  frame_len=frame_length,
                                                  frame_step=frame_step)
                labels, correct_phones, correct_indices, framecount = db2labels(n_frames,
                                                                                dbfile,
                                                                                phonelist,
                                                                                framecount)
                new_indata = np.take(indata, indices=correct_indices, axis=0)
                new_labels = np.take(labels, indices=correct_indices, axis=0)
                npzfilename = f'{dest_flp}{spkr}_{utter}.npz'
                np.savez(npzfilename, indata=new_indata, labels=new_labels)
                if trainOtest:
                    traincount += len(correct_indices)
                else:
                    testcount += len(correct_indices)
                word2phonesdict[f'{spkr}_{utter}'] = word.split('_')
        del phonelist[-2:]
        phonelist.append([traincount,testcount])
        with open(selected_phones_name,'wb') as wp:
            pk.dump(phonelist,wp)
        with open(word2phonesdict_name,'wb') as wp:
            pk.dump(word2phonesdict,wp)
    else:
        print('Already Have Train_Select & Test_Select sets...')
        with open(selected_phones_name,'rb') as rp:
            phonelist = pk.load(rp)
        with open(word2phonesdict_name,'rb') as rp:
            word2phonesdict = pk.load(rp)

    return phonelist[:-1],phonelist[-1],word2phonesdict


""" Check dbs for continuous phone labels"""
def dbcontlabelcounts2dict(dbfile,contphone_dict):
    with open(dbfile,'r',encoding='utf-8')as f:
        lines = [l.split() for l in f.readlines()]
    contphone_list = []
    for line in lines[1:-1]:
        if len(line) >= 4:
            tstart,tend, phone = line[:3]
            contphone_list.append(phone)
        else:
            return contphone_dict
    contphone = '_'.join(p for p in contphone_list)
    if contphone in contphone_dict:
        contphone_dict[contphone] += 1
    else:
        contphone_dict[contphone] = 1
    return contphone_dict

""" get word counts from db corpus"""
def surveydb2contdict(db_corpus,wordcount,internal=False):
    contphone_dict = defaultdict(int)
    totalwordcount = 0
    selectwordlist = []
    for subdir, dirs, files in os.walk(db_corpus):
        dirs.sort()
        for file in sorted(files):
            if file.endswith('.db'):
                totalwordcount += 1
                contphone_dict = dbcontlabelcounts2dict(subdir+'/'+file,contphone_dict)
    print(f' These are the continous words (in dict):{contphone_dict}')
    for w in sorted(contphone_dict,key=contphone_dict.get,reverse=True):
        if contphone_dict[w] >= wordcount:
            selectwordlist.append(w)
    wordadd_dict = defaultdict(list)
    print(f' Selected words (amount {len(selectwordlist)}), by dbdir, {selectwordlist}')
    for subdir, dirs, files in os.walk(db_corpus):
        dirs.sort()
        speaker = subdir.split('/')[-1]
        for file in sorted(files):
            if file.endswith('.db'):
                fadd = subdir+'/'+file
                key = [k for k in dbcontlabelcounts2dict(fadd,{}).keys()]
                if key:
                    if key[0] in selectwordlist:
                        wordadd_dict[key[0]].append(speaker+'_'+file)
    if not internal:
        for k,v in wordadd_dict.items():
            print(k,v)
    return selectwordlist,wordadd_dict


def main():
    Frame_length = 0.025  # 25 ms width
    Frame_step = 0.01  # 10 ms step
    N_ceps = 26
    N_context = 2
    wordcount = 5
    SysPath = os.getcwd().split('GOP-LSTM')[0]
    Child_dat = SysPath + 'corpus/dat/speakers/'
    Dbdir = SysPath + 'GOP-LSTM/PhoneInfo/speakers_db_correct/'
    Holddir = SysPath + 'HoldDir/'
    Phoinfdir = SysPath + 'GOP-LSTM/PhoneInfo/'
    print(f'Wavdir:{Child_dat}  dbdir:{Dbdir}')

    createTT = False
    if createTT:
        selected, totalcount,w2pdict = createNcount_trainNtest(frame_length=Frame_length,
                                                               frame_step=Frame_step,
                                                               n_ceps=N_ceps,
                                                               n_context=N_context,
                                                               dbdir=Dbdir,
                                                               datdir=Child_dat,
                                                               wordcount=wordcount,
                                                               holddir=Holddir,
                                                               phoinfdir=Phoinfdir,
                                                               overwrite=True)
        print(f'Selected:{selected}')
        print(f'Number of Selected:{len(selected)}')
        print(f'Total counts [Train,Test]: {totalcount}')
        print()
    else:
        swl, wad = surveydb2contdict(Dbdir,wordcount,internal=True)

        '''Print to check if we can get similar words in corpus, at 10 we are able to get two
        words that have two or more cases, 9 (ball) and 44 (drum)'''
        origindict = defaultdict(list)
        for ks,va in wad.items():
            varoot = va[0].split('_')[-1].split('.')[0]
            origindict[varoot].append(ks)
        for k,v in origindict.items():
            print(k,v)


if __name__=="__main__":
    main()
