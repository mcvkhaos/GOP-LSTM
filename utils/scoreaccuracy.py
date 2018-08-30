import numpy as np

#gstd = [selected_phones[sp] for sp in np.argmax(y, axis=2)[0]]
#softmax = [selected_phones[sp] for sp in np.argmax(predictions, axis=2)[0]]

def argmaxpredicts(predicts,axis):
    return np.argmax(predicts,axis=axis)

def argmaxpredicts2phones(predicts,select_phones):
    return [select_phones[sp] for sp in argmaxpredicts(predicts,axis=1)]

#forceda = [selected_phones[pind[sp]] for sp in np.argmax(predictions[:, :, pind][0], axis=1)]
def argmaxpredicts2forcedphones(predicts,select_phones,fphones,fwords=False):
    if fwords:
        return np.sum(np.amax(predicts[:, fphones],axis=1)), [select_phones[fphones[sp]] for sp in argmaxpredicts(predicts[:,fphones],axis=1)]
    else:
        return [select_phones[fphones[sp]] for sp in argmaxpredicts(predicts[:,fphones],axis=1)]

