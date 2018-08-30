import os
import codecs
from collections import defaultdict
import numpy as np


def extractTGintervals(targetfile,name):
    start = False
    phone_alignment = []
    try:
        with codecs.open(targetfile, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
    except:
        with codecs.open(targetfile, 'r', encoding='utf-16') as rf:
            lines = rf.readlines()
    if lines:
        for line in lines:
            tokens = line.split()
            if tokens:
                if start:
                    if 'xmin' in tokens[0]:
                        phone_alignment.append([tokens])
                    elif 'intervals' not in tokens[0]:
                        phone_alignment[-1].append(tokens)
                if 'name' in tokens[0]:
                    if tokens[2] == name:
                        start = True
    else:
        print(f'Cannot open file: {targetfile} properly')
        return phone_alignment
    return phone_alignment

def if2eq1(a,b,c):
    if a == c and b == c:
        return True
    else:
        return False

def str2floatdiff(a,b):
    return abs(float(a) - float(b))

def compareTGintervals(source, created, by_utt=False):
    s_uttlen = len(source)
    c_uttlen = len(created)
    if not s_uttlen == c_uttlen and by_utt:
        print(f'Different sizes: {s_uttlen} {c_uttlen}')
    if s_uttlen == 0 or c_uttlen == 0:
        return [],0
    silences = ['"sil"','".pau"','""']
    utt_len_diff = abs(float(source[0][-1][-1]) - float(created[0][-1][-1]))
    if utt_len_diff > 0 and by_utt:
        print(f'Different utterance lengths: {utt_len_diff}')
    utterance = []
    totaldiff = 0
    for x,y in zip(source[1:], created[1:]):
        segdiff = 0
        for a,b in zip(x,y):
            if not if2eq1(a[0],b[0],'text'):
                segdiff += str2floatdiff(a[-1], b[-1])
            else:
                if a[-1] == b[-1]:
                    utterance.append([a[-1]])
                else:
                    if a[-1] in silences and b[-1] in silences:
                        utterance.append(['"sil'])
                    else:
                        utterance.append([a[-1],b[-1]])

        utterance[-1].append(round(segdiff,4))
        totaldiff += round(segdiff,4)
    return utterance, totaldiff

def main():
    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    wordpron_dpath = SysPath + f'HoldDir/MFA_aligns/'
    child_dat = SysPath + 'corpus/dat/speakers/'
    ov_avg = 0
    ov_std = 0
    k = 0
    word_by_word = False
    wordscore_dict = defaultdict(list)

    for subdir, dirs, files in os.walk(wordpron_dpath):
        for file in files:
            fname = file.split('.')
            if fname[1] == 'TextGrid':
                fname2 = fname[0].split('_')
                if fname2[1].isnumeric():
                    targetfile = subdir + '/' + file
                    tgitls_created = extractTGintervals(targetfile,'"phones"')
                    sourcefile = f'{child_dat}{fname2[0]}/{fname2[1]}.TextGrid'
                    tgitls_source = extractTGintervals(sourcefile,'"IPA"')
                    utt,totaldiff = compareTGintervals(source=tgitls_source,
                                                       created=tgitls_created,
                                                       by_utt=word_by_word)
                    if fname2[1] not in wordscore_dict:
                        wordscore_dict[fname2[1]] = [totaldiff]
                    else:
                        wordscore_dict[fname2[1]].append(totaldiff)

                    if utt and word_by_word:
                        print(f'Utterance differences : {utt}')
                        print(f'Total differences {totaldiff} in seconds \n')

    for word in wordscore_dict.keys():
        avg = np.mean(wordscore_dict[word])
        std = np.std(wordscore_dict[word])
        ocs = len(wordscore_dict[word])
        ov_avg += (avg*ocs)
        ov_std += (std*ocs)
        k += ocs
        print(f'{word} has time difference avg: {avg} \n\t\t\t standard dev: {std} '
              f'\n\t\t\t for {ocs} occurances ')
    ov_avg /= k
    ov_std /= k
    print(f'Overall average time diffence: {ov_avg} \n\t\t  std time difference: {ov_std}')
if __name__ == "__main__":
    main()