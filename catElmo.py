import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder
import sys
import builtins
import os
sys.stdout = open("stdout.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
    
print("Started.../")
# model_dir = Path('./path/of/your/downloaded/catELMo')
weights = '/home/sschauh3/partb/catELMo/datasets/catELMoModel/weights.hdf5'
options = '/home/sschauh3/partb/catELMo/datasets/catELMoModel/options.json'
print("Embedder started")
embedder  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU

def catELMo_embedding(x):

    result = []
    for element in embedder.embed_sentences(x):
        # print("HERERERERERE")
        result.append(torch.tensor(element).sum(dim=0).mean(dim=0).tolist())

    return result

dat = pd.read_csv('/home/sschauh3/partb/catELMo/datasets/BindingAffinityPrediction/TCREpitopePairs.csv')
print("data loaded")
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

print("Embedding...")
print("Started")

dat['epi_embeds'] = catELMo_embedding(dat['epi'].tolist())
dat['tcr_embeds'] = catELMo_embedding(dat['tcr'].tolist())

# dat['epi_embeds'] = dat[['epi']].applymap(lambda x: catELMo_embedding(x))['epi']
# dat['tcr_embeds'] = dat[['tcr']].applymap(lambda x: catELMo_embedding(x))['tcr']
print("Writing and saving to file")

dat.to_pickle("/home/sschauh3/partb/catELMo/output/output_new_cuda_vanilla_baseline/data.pkl")   
print("Ended")
