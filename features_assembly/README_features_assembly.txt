features_assembly contains the following scripts:
	1. make_features contains:
		a. speakers_trainNtest - main.program that creates Train & Test directories and fills them with mfcc+label numpy arrays at FRAME-LEVEL
		b. wav2mfcc(invec/inmat) - creates mfccs from wav in vector/matrix forms
		c. db2labels - returns correct/incorrect label list
        d. phones2list - all phone list, saved as numpy
    2. create_TrainTest_wordlevel_fromdbcorrect contains:
        a. createNcount_trainNtest - main.program that create T&T directors at WORD-LEVEL
        b. dbcontlabelcounts2dict - count db continous labels
        c. surveydb2contdict - counts same words from db files
    3. select_features.py - counts phones by frames and subsets into select T&T directories
    4. textgrid2db.py - converts textgrids to dbs


History
05-31-18- Prior to this date, each os held its own 'Holddir' and additionaly save its own 'selected_phones' and 'wordphones2dict' which meant they trained their NN with different Train/Test sets. However, most of the train/testing was done on Cluster and thus results still hold relative (to each other) significance.
