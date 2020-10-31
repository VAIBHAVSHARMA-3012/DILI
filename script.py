#!/usr/bin/env python
# coding: utf-8




from tensorflow import keras
import csv
import pickle
import os.path as op
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Torsions





def load_model(filepath):
    if op.exists(filepath) and op.isfile(filepath):
        model = keras.models.load_model(filepath)
        return model
    else:
        print("File doesn't exist or is corrupted.")

def create_descriptor(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    descriptor = AllChem.GetMorganFingerprintAsBitVect(mol,2)
    return np.reshape(descriptor, (1,32, 64))



if __name__ == '__main__':
    
    filepath="CNN-DILI.h5"
    #smiles="CC(O)=O.[H][C@@]12CCC3=CC(=CC=C3[C@@]1(C)CCC[C@@]2(C)CN)C(C)C"
    model = load_model(filepath)
    #print("Model Loaded.")
    fingerprint = create_descriptor(smiles)
    #print("Descriptor Created.")
    pred = model.predict(fingerprint)
    #print("Predicted Classes")
    print(pred.argmax(axis=1))






