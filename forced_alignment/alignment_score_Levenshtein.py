import os
import codecs
from collections import defaultdict
import numpy as np
import Levenshtein as lv


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


def extractDbintervals(targetfile):
    with open(targetfile,'r') as rf:
        readin = [x.split() for x in rf.readlines()]
    return readin


def cleanTGintervals(tg):
    ign_list = ['".pau"','"sp"','".exm"','"sil"','""']
    clean = [s[-1] for seg in tg if len(seg) >= 3 and seg[-1][-1] not in ign_list for s in seg]
    for i in range(2,len(clean),3):
        clean[i] = clean[i].replace('"','')
    return [clean[i:i+3] for i in range(0,len(clean),3)]


def cleanDbintervals(db):
    ign_list = ['.pau','sp','.exm','sil']
    clean = [seg[:3] for seg in db if seg[-1] not in ign_list]
    return clean


def if2eq1(a,b,c):
    if a == c and b == c:
        return True
    else:
        return False


def str2float_diffrnd(a,b,c):
    return round(round(float(a),c) - float(b),c)


def TG2numstring(tg,plist):
    numstring = ''
    for e in tg:
        e = e[-1]
        if e not in plist:
            plist.append(e)
        numstring += chr(plist.index(e)+33)
    return numstring,plist


def uttLD(gstd, syth, phonenum_list, printing=False):
    gstd_string, phonenum_list = TG2numstring(gstd, phonenum_list)
    syth_string, phonenum_list = TG2numstring(syth, phonenum_list)
    answer = lv.distance(gstd_string, syth_string)
    if printing:
        print(num2string(gstd_string, phonenum_list))
        print(num2string(syth_string, phonenum_list))
        print(answer)
    return answer


def num2string(numstring,plist):
    phonestring = ''
    for n in list(numstring):
        phonestring += f' {plist[ord(n)-33]}'
    return phonestring


def eqcompareTGintervals(gstd, sythn):
    utterance = []
    totaldiff = 0
    totaltime = str2float_diffrnd(gstd[-1][-2],gstd[0][0],4)
    start_time = str2float_diffrnd(gstd[0][0],sythn[0][0],4)
    if start_time > 0:
        totaldiff -= start_time
    for x,y in zip(gstd,sythn):
        start_time = str2float_diffrnd(x[0],y[0],4)
        end_time = str2float_diffrnd(x[1],y[1],4)
        utterance.append([x[2],start_time,end_time])
        totaldiff += abs(start_time)
    end_time = str2float_diffrnd(gstd[-1][-2], sythn[-1][-2], 4)
    if end_time > 0:
        totaldiff += end_time
    return utterance, totaldiff, totaltime


def scoreMFA(wordpron_dpath,dbdir,word_by_word=False):
    ov_avg = 0
    ov_std = 0
    k = 0
    total = 0
    insdelsub = 0
    wordscore_dict = defaultdict(list)
    phonenum_list = []
    for subdir, dirs, files in os.walk(wordpron_dpath):
        for file in files:
            fname = file.split('.')
            if fname[1] == 'TextGrid':
                fname2 = fname[0].split('_')
                if fname2[1].isnumeric():
                    total += 1
                    syth_file = subdir + '/' + file
                    syth_tgitl = extractTGintervals(syth_file,'"phones"')
                    gstd_file = f'{dbdir}{fname2[0]}/{fname2[1]}.db'
                    gstd_tgitl = extractDbintervals(gstd_file)
                    gstd_tgitl = cleanDbintervals(gstd_tgitl)
                    syth_tgitl = cleanTGintervals(syth_tgitl)
                    answer = uttLD(gstd_tgitl,syth_tgitl,phonenum_list,printing=False)
                    if answer == 0:
                        utt,totaldiff,totaltime = eqcompareTGintervals(gstd_tgitl,syth_tgitl)
                        if fname2[1] not in wordscore_dict:
                            wordscore_dict[fname2[1]] = [totaldiff]
                        else:
                            wordscore_dict[fname2[1]].append(totaldiff/totaltime)
                        if utt and word_by_word:
                            print(f'Utterance differences : {utt}')
                            print(f'Total differences {totaldiff} in seconds \n')
                    else:
                        insdelsub += 1
    print(f'n files:{len(files)}')
    for word in wordscore_dict.keys():
        avg = np.mean(wordscore_dict[word])
        std = np.std(wordscore_dict[word])
        ocs = len(wordscore_dict[word])
        ov_avg += (avg*ocs)
        ov_std += (std*ocs)
        k += ocs
        if word_by_word:
            print(f'{word} has time difference avg: {avg} \n\t\t\t standard dev: {std} ')
            print(f'\t\t\t for {ocs} occurances ')
    ov_avg /= k
    ov_std /= k
    print(f'Word avg time difference (percent): {round(ov_avg,4)} ')
    print(f'Overall std time difference (percent): {round(ov_std,4)} ')
    print(f'Average word time accuracy: {round(1-ov_avg,4)}  ')
    print(f'Insertions, deletions, substitutions: {insdelsub} out of {total}, {round((insdelsub/total*100),4)}  ')


def main():
    cwd = os.getcwd()
    SysPath = cwd.split('GOP-LSTM')[0]
    dpath = SysPath + f'HoldDir/MFA_aligned/'
    Dbdir = SysPath + 'GOP-LSTM/PhoneInfo/speakers_db_correct/'
    MFAs = os.listdir(dpath)
    for mfa in MFAs:
        print(mfa)
        scoreMFA(wordpron_dpath=dpath+mfa,dbdir=Dbdir,word_by_word=False)
        print('\n')

if __name__ == "__main__":
    main()