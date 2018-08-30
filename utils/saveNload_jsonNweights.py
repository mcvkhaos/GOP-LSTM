from keras.models import model_from_json
import os
def save_json(model, model_json_address):
    model_json = model.to_json()
    with open(model_json_address, 'w') as jf:
        jf.write(model_json)

def load_json(model_json_address):
    if os.path.exists(model_json_address):
        with open(model_json_address,'r') as jf:
            return model_from_json(jf.read())
    else:
        print(f'Missing model file: {model_json_address}' )

def take_Nweights(n_layers,modelwithweights,modelrecievingweights):
    for n in range(n_layers):
        modelrecievingweights.layers[n].set_weights(modelwithweights[n])
    return modelrecievingweights