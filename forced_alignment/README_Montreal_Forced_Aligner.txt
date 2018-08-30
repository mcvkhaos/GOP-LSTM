Download or Install MFA at http://montreal-forced-aligner.readthedocs.io/en/latest/

To train and align, run the following command on a terminal:
~/montreal-forced-aligner/bin/mfa_train_and_align ~/corpus_directory ~/dict_path ~/output_directory

For more info:
~/montreal-forced-aligner/bin/mfa_train_and_align -h

Train and align Example:
~/bin/mfa_train_and_align ~/GOP-LSTM/forced_alignment/Example_Corpus \
~/GOP-LSTM/forced_alignment/Example_Corpus/words.dict ~/GOP-LSTM/forced_alignment/Example_Aligns \
 -t ~/GOP-LSTM/forced_alignment/Example_Model -o ~/GOP-LSTM/forced_alignment/Model.zip

OR: cmd corpus dict alignedfiles -t modeldir - o zippedmodeldir

You'll need the "Model.zip" to use your trained MFA, otherwise you can use one of their pre-trained models.
 For info on align:
 ~/bin/mfa_align -h

