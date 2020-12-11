import torch
import spacy
import json
import pandas as pd

trainde =pd.DataFrame(open('data/train.de','r',encoding='UTF-8').read().strip().split('\n'),columns=["de"])
valde = pd.DataFrame(open('data/val.de','r',encoding='UTF-8').read().strip().split('\n'),columns=["de"])
testde = pd.DataFrame(open('data/test2016.de','r',encoding='UTF-8').read().strip().split('\n'),columns=["de"])
trainen = pd.DataFrame(open('data/train.en','r',encoding='UTF-8').read().strip().split('\n'),columns=["en"])
valen =pd.DataFrame( open('data/val.en','r',encoding='UTF-8').read().strip().split('\n'),columns=["en"])
testen = pd.DataFrame(open('data/test2016.en','r',encoding='UTF-8').read().strip().split('\n'),columns=["en"])

train = pd.concat([trainde,trainen],axis=1)
valid = pd.concat([valde,valen],axis=1)
test = pd.concat([testde,testen],axis=1)


# cf: https://github.com/omarsar/pytorch_neural_machine_translation_attention/blob/master/NMT_in_PyTorch.ipynb