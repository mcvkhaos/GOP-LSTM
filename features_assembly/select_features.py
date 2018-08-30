# make train and test sets for NN classifier
import numpy as np
import pickle as pk
import os


""" Select Phones  for Train Set & Test Set"""
def select_trainNtest(bycount=2000, listofphones=[],holddir='',train_corpus='Train_Correct/',test_corpus='Test_Correct/', overwrite=False):
    frame_count_name = f'framecounts.pk'
    # Parition into train & test
    if os.path.exists(holddir+frame_count_name):
        with open(holddir+frame_count_name,'rb') as rp:
            frame_counts = pk.load(rp)
            print(f'Frames Counts:{frame_counts}')
    else:
        print('Missing Frame Counts!')
    selected_phones_name = f'selected_phones_{bycount}.pk'
    train_select_name = f'Train_Select_{bycount}'
    test_select_name = f'Test_Select_{bycount}'
    # Create Train/Test sets if not already created or if overwrite
    if not overwrite and os.path.exists(holddir+selected_phones_name) and os.path.isdir(holddir+train_select_name) and os.path.isdir(holddir+test_select_name):
        with open(holddir+selected_phones_name,'rb') as rp:
            selected_phones = pk.load(rp)
        print(f'Loading previous select phones list {selected_phones}')
        return selected_phones[:-1],selected_phones[-1]
    else:
        print('Creating Train_Select & Test_Select sets...')
        if not os.path.isdir(holddir+train_select_name):
            os.mkdir(holddir+train_select_name)
        else:
            print('Train_Select already exists. You may want to delete that folder first.')
        if not os.path.isdir(holddir+test_select_name):
            os.mkdir(holddir+test_select_name)
        else:
            print('Test_Select already exists. You may want to delete that folder first.')
        selected_phones = []
        # Select phones in frames_counts based on listofphones or bycount
        if listofphones:
            for phone in listofphones:
                print(f'Selecting {phone} : {frame_counts[phone]}')
                selected_phones.append(phone)

        if bycount:
            print(f'Count Threshold: {bycount}')
            for phone,count in frame_counts.items():
                if count >= bycount:
                    print(f'Selecting {phone} : {frame_counts[phone]}')
                    if phone not in selected_phones:
                        selected_phones.append(phone)

        print(f'Selected Phones:{selected_phones}')
        #Translate Phones into Indices:
        phonelist = np.load(holddir+'phonelist.npy').tolist()
        selected_indices = []
        for phone in selected_phones:
            selected_indices.append(phonelist.index(phone))
        total_trainNtest_count= [0,0]
        print(train_corpus,test_corpus)
        for I,corpus in enumerate([train_corpus,test_corpus]):
            for subdir, dirs, files in os.walk(corpus):
                for file in files:
                    npzfiles = np.load(corpus + file)
                    corp = corpus.split('_')[0]
                    f_prefix = file.split('.')[0]
                    featuresNlabels_file = f'{corp}_Select_{bycount}/{f_prefix}'
                    indata = npzfiles['indata']
                    labels = npzfiles['labels']
                    # Search for correct phones and save them, keep file name if any
                    y_prev = -1
                    indices = []
                    new_labels = []
                    add_to_selected = False
                    for index,y in enumerate(labels):
                        if y == y_prev:
                            if add_to_selected:
                                indices.append(index)
                                new_labels.append(selected_indices.index(y))
                        else:
                            y_prev = y
                            add_to_selected = False
                            if y in selected_indices:
                                add_to_selected = True
                                indices.append(index)
                                new_labels.append(selected_indices.index(y))

                    if indices:
                        selected_indata = np.take(indata,indices=indices,axis=0)
                        np.savez(featuresNlabels_file,
                                 indata=selected_indata,
                                 labels=new_labels)
                        total_trainNtest_count[I] += len(indices)
        print(f'Train & Test sizes: {total_trainNtest_count}')
        selected_phones.append(total_trainNtest_count)
        with open(holddir+selected_phones_name,'wb') as wp:
            pk.dump(selected_phones,wp)
        return selected_phones[:-1], selected_phones[-1]


def main():
    """ Test Component Functions"""
    testing = True
    if testing:
        select_trainNtest(bycount=4000,
                          holddir='~/HoldDir/',
                          train_corpus='~/HoldDir/Train_Correct/',
                          test_corpus='~/HoldDir/Test_Correct/',
                          listofphones=[],
                          overwrite=True)  # Test overall

if __name__ == "__main__":
    main()
