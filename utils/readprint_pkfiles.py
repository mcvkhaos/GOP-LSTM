""" read all pickle files in target folder, specifically reading through dictionaries and lists"""
import os
import pickle as pk

def readprint_dictOlist(dict_path):
    with open(dict_path, 'rb') as rf:
        loaded = pk.load(rf)

    # check if its list or dict
    ctype = type(loaded)
    print(f'Loaded type is: {ctype}')
    if ctype is dict:
        for k,v in loaded.items():
            print(f'Key:{k}')
            print(f'Values:{v}')
    elif ctype is list:
        for index,item in enumerate(loaded):
            print(f'Index:{index}')
            print(f'Item:{item}')
    else:
        print(f'Else:{loaded}')

def scanNchoosefiles2read(folder_path):
    os.chdir(folder_path)
    localfiles = os.listdir(os.curdir)
    print(localfiles)
    for lf in localfiles:
        print(f'\nCurrent file:{lf}')
        if not lf.endswith('.pk'):
            print('\t skipping')
        else:
            print(f'Input any key to read:{lf}')
            print(' otherwise input \'n\'')
            choice = input()
            if choice is not 'n':
                readprint_dictOlist(lf)



cwd = os.getcwd()
SysPath = cwd.split('GOP-LSTM')[0]
Holddir = SysPath + 'HoldDir/'
Phonedir = SysPath + 'GOP-LSTM/PhoneInfo/'

scanNchoosefiles2read(Phonedir)

#
# d_path = Holddir + 'selected_phones.pk'
# readprint_dictOlist(dict_path=d_path)