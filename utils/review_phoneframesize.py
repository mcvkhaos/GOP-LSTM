''' Corpus summary of phone-frame counts, selected phones, number of samples, length of continuous samples, mean/std of length'''
import os
import sys
SysPath = os.getcwd().split('src')[0]  # required to find local packages
sys.path.append(SysPath)

from features_assembly.select_features import select_trainNtest
import numpy as np
import matplotlib.pyplot as plt

# generator consider one file open to be one sample, even though segmented
def nseqsofnsize(traindir,seq_size):
    '''
    files = os.listdir(traindir)
    seqofnframes = 0
    for file in files:
        npzfile = np.load(traindir+file)
        y = npzfile['labels']
        y = np.asarray(y)
        n_frames = len(y)
        seqofnframes += np.ceil(n_frames/seq_size)
    '''
    return len(os.listdir(traindir)) # works during testing

def main():

    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    Holddir = SysPath + 'HoldDir/'
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
    print(f'Total count of samples:{totalcount}')

    files = os.listdir(Traindir)
    maxN_frames = 0
    lengths = []
    seqofn = 0
    seq_size = 16
    for file in files:
        npzfile = np.load(Traindir + file)
        y_train = npzfile['labels']
        y_train = np.asarray(y_train)
        n_frames = len(y_train)
        seqofn += int(np.floor(n_frames/seq_size))
        lengths.append(n_frames)
        if maxN_frames < n_frames:
            maxN_frames = n_frames
    lengths = np.asarray(lengths)
    u = np.mean(lengths)
    sd = np.std(lengths)
    print(f'Max number of frames:{maxN_frames}')
    print(f'Mean: {u} StandDev: {sd}')
    print(f'{seqofn} sequences of {seq_size} length')
    if False:
        plt.hist(lengths)
        plt.show()

    print(nseqsofnsize(Traindir,seq_size))


if __name__ == "__main__":
    main()
