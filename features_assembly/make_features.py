# make train and test sets for NN classifiers
import scipy.io.wavfile as wav
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
import numpy as np
import pickle as pk
import os
from utils.oddsNends import ifnodirmkdir_elsewarn


""" Converts Wav File to MFCCS to produce input vector """
def wav2mfcc2invec(wavfile, n_ceps, n_context, frame_len=0.025, frame_step=0.01):
    freqrate, values = wav.read(wavfile)
    # http://python-speech-features.readthedocs.io/en/latest/
    mfccs = mfcc(values, samplerate=freqrate, numcep=n_ceps, winlen=frame_len, winstep=frame_step,nfft=2048)
    n_frames = mfccs.shape[0]
    n_features = n_frames*n_ceps+n_ceps
    inputsWcontext = np.empty((n_frames, n_features), dtype=np.float64)

    # fill edge inputs
    if n_context > 0:
        for n in range(n_context + 1):
            inputsWcontext[n, :] = 0
            inputsWcontext[n, (n_context - n) * n_ceps:] = mfccs[:n + n_context + 1, :].flatten()
        for n in range(n_context + 1, 0, -1):
            index = n_frames - n
            inputsWcontext[index, :] = 0
            inputsWcontext[index, :n * n_ceps] = mfccs[index:, :].flatten()

    # fill the bulk
    for n in range(n_context, n_frames - n_context):
        inputsWcontext[n, :] = mfccs[n - n_context:n + n_context + 1, :].flatten()
    print(inputsWcontext.shape)
    return inputsWcontext, n_frames


""" Converts Wav File to MFCCS to produce input matrix """
def wav2mfcc2inmat(wavfile, n_ceps, n_context, frame_len=0.025, frame_step=0.01):
    freqrate, values = wav.read(wavfile)
    # http://python-speech-features.readthedocs.io/en/latest/
    mfccs = mfcc(values, samplerate=freqrate, numcep=n_ceps, winlen=frame_len, winstep=frame_step,nfft=2048)
    mfccs /= 255.
    n_samples = mfccs.shape[0]
    n_frames = 2 * n_context + 1
    inputsWcontext = np.empty((n_samples,n_frames, n_ceps,1), dtype=np.float64)

    # fill edge inputs
    if n_context > 0:
        for n in range(n_context + 1):
            inputsWcontext[n,:, :] = 0
            inputsWcontext[n, n_context - n:, :, 0] = mfccs[:n + n_context +1 , :]
        for n in range(n_context + 1, 0, -1):
            index = n_samples - n
            inputsWcontext[index, :, :] = 0
            inputsWcontext[index, :n, :, 0] = mfccs[index:, :]

    # fill the bulk
    for n in range(n_context, n_samples - n_context):
        inputsWcontext[n, :, :, 0] = mfccs[n - n_context:n + n_context + 1, :]
    return inputsWcontext, n_samples


""" Uses db to produce labels and correct/incorrect label list """
def db2labels(n_frames, dbfile,phonelist,framecount):
    labels = []
    correct_indices = []
    correct_phones = []
    with open(dbfile,'r',encoding='utf-8')as f:
        lines = [l.split() for l in f.readlines()]
    stepsize = float(lines[-1][1]) / n_frames
    tstart = 0.0
    for line in lines:
        correct_phone = False
        if len(line) >= 4:
            _,tend, phone = line[:3]
            correct_phone = True
            correct_phones.append(phone)
        elif len(line) == 3:
            _, tend, phone = line
        elif len(line) == 2:
            _,tend = line
            phone = '.pau'
        else:
            print(f'ERROR db label at {dbfile}, length is {len(line)}')
            break
        n_labels = round((np.float(tend) - tstart) / stepsize)
        start = len(labels)
        end = start + n_labels
        if correct_phone:
            start = len(labels)
            end = start + n_labels
            correct_indices.append([start,end])
        labels += n_labels * [phonelist.index(phone)]
        tstart = np.float(tend)
    # rounding error requires +/- values
    n_labels = len(labels)
    if n_frames > n_labels:
        n_diff = n_frames - n_labels
        labels += n_diff * [phonelist.index(phone)]
        if correct_phone:
            correct_indices[-1][1] += (n_diff)
    elif n_frames < n_labels:
        n_diff = n_labels - n_frames
        labels = labels[:n_labels - n_diff]
        if correct_indices:
            if correct_phone or correct_indices[-1][1] > n_frames:
                correct_indices[-1][1] -= (n_diff)


    for index,(phone,indices) in enumerate(zip(correct_phones,correct_indices)):
        if phone in framecount:
            framecount[phone] += (indices[1]-indices[0])
        else:
            framecount[phone] = (indices[1] - indices[0])
        correct_indices[index] = [x for x in range(indices[0],indices[1])]
    correct_indices_list = [i for l in correct_indices for i in l]

    return labels, correct_phones, correct_indices_list,framecount


""" Reads all phones in data(db) and creates a List"""
def phones2list(holddir,corpus, overwrite=False):
    if not os.path.exists(holddir+'phonelist.npy') or overwrite:
        wf = open(holddir+'phonelist.txt', 'w')
        wf.write(f'.pau  0 \n')
        k = 1
        phonelist = ['.pau']
        for subdir, dirs, files in os.walk(corpus):
            for file in files:
                if file.endswith('.db'):
                    with open(subdir + '/' + file, 'r') as f:
                        for line in f:
                            values = line.split()
                            phone = values[-1]
                            if phone not in phonelist and len(values) > 2:  # db.py not working
                                phonelist.append(phone)
                                wf.write(f'{phone}  {k} \n')
                                k += 1
        np.save(holddir+'phonelist.npy', phonelist)
        wf.close()

    else:
        phonelist = np.load(holddir+'phonelist.npy').tolist()
    return phonelist


""" Create Train Set & Test Set for separate speakers"""
def speakers_trainNtest(db_corpus, wav_corpus, n_ceps, n_context,holddir,correct=True,frame_length=0.025,frame_step=0.01,inmat=True, overwrite=False):
    if correct:
        traindir = holddir + 'Train_Correct'
        testdir = holddir + 'Test_Correct'
        all_mfccs = holddir + 'ALL_MFCCs_Correct'
    else:
        traindir = holddir + 'Train'
        testdir = holddir + 'Test'
        all_mfccs = holddir + 'ALL_MFCCs'

    phonelist = phones2list(holddir,db_corpus,overwrite=True)
    framecount = {}
    if inmat == True:
        infunc = wav2mfcc2inmat
    else:
        infunc = wav2mfcc2invec
    # Parition into train & test
    if correct:
        print('Creating Train_Correct & Test_Correct sets...')
    else:
        print('Creating Train & Test sets...')

    if not os.path.isdir(traindir) or not os.path.isdir(testdir) or overwrite:
        ifnodirmkdir_elsewarn(traindir)
        ifnodirmkdir_elsewarn(testdir)
        ifnodirmkdir_elsewarn(all_mfccs)

        allspeakers = os.listdir(db_corpus)
        x_train, x_test = train_test_split(allspeakers, test_size=0.1, random_state=7)
        for subdir, dirs, files in os.walk(db_corpus):
            cdir = subdir.split('/')[-1]
            if cdir in x_test:
                destdir = testdir
            elif cdir in x_train:
                destdir = traindir
            else:
                continue
            for file in files:
                f_prefix = file.split('.')[0]
                featuresNlabels_file = f'{destdir}/{cdir}_{f_prefix}_features_labels'
                featuresNlabels_file_MFCCs = f'{all_mfccs}/{cdir}_{f_prefix}_features_labels'
                if not os.path.exists(featuresNlabels_file+".npz") or overwrite:
                    wav_file = f'{wav_corpus}{cdir}/{f_prefix}.wav'
                    db_file = f'{db_corpus}{cdir}/{file}'
                    indata, n_frames = infunc(wavfile=wav_file,
                                              n_ceps=n_ceps,
                                              n_context=n_context,
                                              frame_len=frame_length,
                                              frame_step=frame_step)
                    labels, correct_phones, correct_indices,framecount = db2labels(n_frames=n_frames,
                                                                                   dbfile=db_file,
                                                                                   phonelist=phonelist,
                                                                                   framecount=framecount)
                    new_indata = np.take(indata, indices=correct_indices, axis=0)
                    new_labels = np.take(labels, indices=correct_indices, axis=0)
                    if new_indata.shape[0] != 0:
                        np.savez(featuresNlabels_file,indata=new_indata,labels=new_labels)
                    np.savez(featuresNlabels_file_MFCCs, indata=indata, labels=labels)
        with open(holddir+'Correct_Frame_Counts', 'w') as wf:
            for keys,values in framecount.items():
                wf.write(f'{keys} {values} \n')
        with open(holddir+'framecounts.pk','wb') as wp:
            pk.dump(framecount,wp)


def main():
    """ Config Values for testing or manually running speakers_trainNtest """
    Frame_length = 0.025  # 25 ms width
    Frame_step = 0.01  # 10 ms step
    Dbdir = '~/GOP-LSTM/src/speakers_db_correct/'
    Wavdir = '~corpus/dat/speakers/'  ## set for you system
    Holddir = '~/HoldDir/'
    N_ceps = 26
    N_context = 2

    """ Test Component Functions """
    testing = False
    if testing:
        Dbfile = './speakers_db_correct/O-33/23.db'
        Wavfile = '~corpus/dat/speakers/O-33/23.wav'
        inmat, n_frames = wav2mfcc2inmat(Wavfile, n_ceps=13, n_context=2)
        phonelist = phones2list(Dbdir, overwrite=True)
        framecount = {}
        labels, correct_phones, correct_indices, framecount = db2labels(n_frames, Dbfile, phonelist, framecount)
        print(framecount)
        print(inmat.shape)
        print(correct_indices)
        print(correct_phones)
        new_inmat = np.take(inmat, indices=correct_indices, axis=0)
        new_labels = np.take(labels, indices=correct_indices, axis=0)
        print(new_inmat.shape, new_labels)

    if not testing:
        speakers_trainNtest(db_corpus=Dbdir,
                            wav_corpus=Wavdir,
                            n_ceps=N_ceps,
                            n_context=N_context,
                            holddir=Holddir,
                            overwrite=True)  # Test overall
if __name__ == "__main__":
    main()
