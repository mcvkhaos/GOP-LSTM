import os
from forced_alignment.alignment_TextGrid.tg import TextGrid
# Check file by file, the alignment accuracy
index = 1
cwd = os.getcwd()
SysPath = cwd.split('GOP-LSTM')[0]
wordpron_dpath = SysPath + f'HoldDir/MFA_aligns/'

child_dat = SysPath + 'corpus/dat/speakers/'

for subdir, dirs, files in os.walk(wordpron_dpath):
    for file in files:
        fname = file.split('.')
        print(fname)
        if fname[1] == 'TextGrid':
            fname2 = fname[0].split('_')
            if fname2[1].isnumeric():
                tfile = subdir + '/' + file
                t = TextGrid(tfile,['v'])
                print(t)
        break
