utils contains the following:
    1. generators_multifiles - has an assortment of generator for specific NNA (Single vs Batch vs Sequence)
    2. model_selection - saved model info, hardly used
    3. attention_decoder - from https://github.com/datalogue/keras-attention
    4. oddsNends - odds and ends (random reused functions)



History:
05-28-18 - The generator for seq2seq model wasn't including batches missing values, ex. batches with 56 out of 64 didn't get used. I've changed that now to include blanks, '_'. Its already improved performance. However, the previous results cannot be directly used, but relatively they're useful in finding the best performers.