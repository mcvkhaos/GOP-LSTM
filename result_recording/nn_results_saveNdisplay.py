import os
import pandas as pd
import pickle as pk
from collections import defaultdict


def results_address_check(results_address):
    if results_address is None:
        return os.getcwd() + '/'
    else:
        return results_address


def dict_loadOnew(dict_address, overwrite=False):
    if os.path.exists(dict_address) and not overwrite:
        with open(dict_address,'rb') as rp:
            master_records = pk.load(rp)
        print(f'Loading...{dict_address}')
        return master_records
    else:
        return defaultdict(list)


def dict_save(dict_address, record_dict):
    with open(dict_address, 'wb') as wp:
        pk.dump(record_dict,wp)


def nnrecords_add(html_corpus,nn_record_name, results_address=None,overwrite=False):
    results_address = results_address_check(results_address)
    nn_records_address = results_address + nn_record_name
    master_records = dict_loadOnew(nn_records_address, overwrite)
    for _,_,files in os.walk(html_corpus):
        for file in files:
            name = str(file.split('/')[-1].split('.')[0])
            if name not in master_records:
                x = pd.read_html(html_corpus+file)[0]
                temp = [x['mean_test_score'].max(),x['params'][x['mean_test_score'].idxmax()]]
                master_records[name] = temp
            else:
                print(f'NN: {name} already saved')
    dict_save(nn_records_address, master_records)


def nnrecords_rankNprint(nn_record_name, results_address=None,overwrite=False):
    results_address = results_address_check(results_address)
    nn_record_address = results_address + nn_record_name
    master_records = dict_loadOnew(nn_record_address,overwrite)
    for i in sorted(master_records.items(),key=lambda  x:x[1],reverse=True):
        print(i)


def cdnn_records_add(loss, accuracy, model_name, nn_records_name, results_address=None, overwrite=False):
    results_address = results_address_check(results_address)
    nn_records_address = results_address + nn_records_name
    master_records = dict_loadOnew(nn_records_address, overwrite)

    if model_name not in master_records:
        master_records[model_name] = [(loss,accuracy)]
    else:
        master_records[model_name].append((loss,accuracy))
        print(f'NN: {model_name} already saved')

    dict_save(nn_records_address, master_records)


def cdnn_records_rmzerosNwithouttests(nn_record_name, results_address=None):
    results_address = results_address_check(results_address)
    nn_records_address = results_address + nn_record_name
    master_records = dict_loadOnew(nn_records_address)
    delete_list = []
    for key in master_records.keys():
        if master_records[key][0][0] == 0 and master_records[key][0][1] == 0:
            print(f'Removing: {master_records[key]} at  {key}')
            delete_list.append(key)
        if len(master_records[key]) < 2:
            print(f'Removing: {key} with test cases {len(master_records[key])}')
            delete_list.append(key)
    for key in delete_list:
        del master_records[key]

    dict_save(nn_records_address, master_records)


def cdnn_records_rankNprint(nn_record_name, results_address=None,overwrite=False):
    results_address = results_address_check(results_address)
    nn_record_address = results_address + nn_record_name
    master_records = dict_loadOnew(nn_record_address,overwrite)
    for name in sorted(master_records,reverse=True):
        print(name)
        #print(master_records[name])
        #print(len(master_records[name]))
        for i,(x,y) in enumerate(master_records[name][1:]):
            if i == 0:
                print(f'Loss: {x}')
                print(f'Accuracy: {y}')
            if i == 1:
                print(f'Max All: {x}%')
                print(f'ForcedA: {y}%')
        print('\n')


def cdnn_records_sortNplot(nn_record_name, results_address=None,overwrite=False):
    results_address = results_address_check(results_address)
    nn_record_address = results_address + nn_record_name
    master_records = dict_loadOnew(nn_record_address,overwrite)
    for name in sorted(master_records,reverse=True):
        print(name)
        factors = [x for x in name.split('_')]
        print(factors)
        for i,(x,y) in enumerate(master_records[name][1:]):
            if i == 0:
                print(f'Loss: {x}')
                print(f'Accuracy: {y}')
            if i == 1:
                print(f'Max All: {x}')
                print(f'ForcedA: {y}')
        print('\n')


def cdnn_records_check(model_name,nn_records_name, results_address=None):
    results_address = results_address_check(results_address)
    nn_records_address = results_address + nn_records_name
    master_records = dict_loadOnew(nn_records_address, overwrite=False)
    for model in master_records.keys():
        model = model.split('_E')
        if model_name == model[0]:
            return True
    return False


def main():
    GOP-LSTM_dir = os.getcwd().split('GOP-LSTM')[0]
    html_score = False
    if html_score:
        html_dict_name = 'nn_records.pk'
        html_corpus = GOP-LSTM_dir + 'GOP-LSTM/HTML_results/AE_phonemes'
        nnrecords_add(html_corpus=html_corpus,nn_record_name=html_dict_name)
        nnrecords_rankNprint()

    CDNN = True
    if CDNN:
        cdnn_dict_name = 'cdnn_gridsearch_records_wl30.pk'
        cdnn_dict_name = 'cp_blstm_gridsearch_records_wl_30.pk'
        cdnn_dict_name = 'dcp_blstm_gridsearch_records_wl_25.pk'
        cdnn_dict_name = 'ddrescp_blstm_gridsearch_records_wl_25.pk'
        #cdnn_dict_name = 'cp_attentlstm_gridsearch_records_wl_30.pk'
        #cdnn_dict_name = 'crnn_gridsearch_records_4000.pk'
        #cdnn_dict_name = 'crnn_testing_4000.pk'
        cdnn_address = GOP-LSTM_dir + '/GOP-LSTM/Results/CDNN_phones/'
        listofaddresses = os.listdir(cdnn_address)
        for address in listofaddresses:
            cdnn_records_rmzerosNwithouttests(address,cdnn_address)
            cdnn_records_rankNprint(nn_record_name=address,results_address=cdnn_address)


    testing = False
    if testing:
        cdnn_dict_name = 'testing_records.pk'
        cdnn_address = GOP-LSTM_dir + '/GOP-LSTM/Results/CDNN_phones/'
        cdnn_records_add(loss=4,accuracy=20,model_name='ex',nn_records_name=cdnn_dict_name,results_address=cdnn_address)
        cdnn_records_rankNprint(nn_record_name=cdnn_dict_name, results_address=cdnn_address)


if __name__ == "__main__":
    main()
