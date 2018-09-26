# GOP-LSTM
This repo is the code base of "Improving the Goodness of Pronunciation score by using Deep Neural Networks: Single-input Classification and Sequence-to-Sequence Classification" (2018). 

The OHSU Child Corpus is not included. 

#### GOP-LSTM is organized into the following folders:
    1. Results
    2. features_assembly
    3. force_alignment
    4. result_recording
    5. src
    6. utils
    7. External: HoldDir

##### Results
holds AE_phones(AutoEncoder) and CDNN_phones(Convolutional & Deep Neural Networks). These two folders hold the saved results from training and testing DNN in the form of Pickled
    Python Dictionaries.

##### features_assembly
contains scripts for creating train and test folders from the Childs' corpus.

##### forced_alignment
holds the Montreal Forced Aligner(MFA) scripts, custom feature_assembly scripts for MFA use, and alignment evaluation.

##### result_recording
uses Results to sort and print results, plot results, or create sorted csv results for LaTex tables.

##### src
contains scripts for creating, training, and testing Neural Network architectures on specified features and sbatch scripts for running on slurm cluster. NOTE: to save Train and Test directories you'll need to setup a "HoldDir" (Hold directory) somewhere on you computer and link in the script file appropriately.

##### utils
contains additional scripts, such as generators for training, model saving/loading/weight transfer, and attention decoder

##### HoldDir
holds Train & Test directories to prevent having to . them with git repeatedly
