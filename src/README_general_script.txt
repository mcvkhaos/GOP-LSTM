src generalized NN architecture (NNA) script:
	1. Set CUDA to appropriate device
	2. Set System Path to current location (required when switching between devices)
	3. Loading necessary packages: keras, tensorflow, and project-made functions
	4. Create NNA basic functions:
		a. make model
		b. train model (internally saves results to set dictionary path)
		c. evalaute model + predict_model ('''')

	5. Main -
		a. Config lines: set Frame_Length, Wav(e)_dir(ectory), to the User's desired values.
		b. Frame-level or Word-level Train & Test feature assembly.
			i. Frame-level has args such as ByCount(frame number threshold)
			ii. Word-level ''''  wordcount(word number threshold)
		c. gridsearch - set parameters lists for NNA, e.g. ConvLayerList, DropList
		d. for loops to run gridsearch
		e. iterate through NNA basic functions: make model, train model, evaluate/predict model
		d. delete model & clear keras session for next NN

Use_* are special case scripts, which are designed to test either
    1. WordMax - Iterate through all words with Word Focus and use sum of softmax for ranking
    2. Same Word Different Pronunciation - compare close words