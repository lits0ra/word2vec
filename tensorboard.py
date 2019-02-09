import gensim
import torch
from tensorboardX import SummaryWriter

vec_path = './model.model'

writer = SummaryWriter()
model = gensim.models.Word2Vec.load(vec_path)
weights = model.wv.vectors
labels = model.wv.index2word

weights = weights[:1000]
labels = labels[:1000]

writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
