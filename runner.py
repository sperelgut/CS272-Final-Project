import readers
from icecream import ic
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

temp_dir = 'D:\ProgramFolder\Repos\\final_project\data'

#### Anecdotes
corpus = readers.ScruplesCorpus(data_dir=temp_dir)



corpus2 = readers.ScruplesCorpusDataset(
            data_dir=temp_dir,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=None)


#### Graph Out Word Distributions ####


#### Graph out text body distribution ####
fig,ax = plt.subplots(1,1)

#print(corpus.train[1]["text"].astype(str).apply(len).max())
text_len=np.zeros(corpus.train[1]["text"].shape,dtype=np.int32)
for i,title in enumerate(corpus.train[1]["text"]):
    text_len[i] =int(len(title.split(" ")))

print(np.mean(text_len))
print("toal:",len(text_len))
print("clipped:",len(text_len[text_len > 750]))
ax.hist(text_len, bins = range(0,2200))
ax.set_title("Text Body Distribution")
plt.show()


#### Graph out title distribution ####

#print(corpus.train[1]["title"].astype(str).apply(len).mean())
lens=np.zeros(corpus.train[1]["title"].shape,dtype=np.int32)
for i,title in enumerate(corpus.train[1]["title"]):
    lens[i] =int(len(title.split(" ")))
#print(lens)

#print(np.mean(lens))
ax.hist(lens, bins = range(0,50))
ax.set_xticks(range(0,50))
ax.set_title("Title Length Distribution")
plt.show()


