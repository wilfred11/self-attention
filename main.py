#https://www.kaggle.com/code/aisuko/coding-the-self-attention-mechanism

#https://medium.com/data-science/contextual-transformer-embeddings-using-self-attention-explained-with-diagrams-and-python-code-d7a9f0f4d94e
import torch

from embed import do_embed
from model import SelfAttention
from visualize import visualise
from weight import define_weights, attention

torch.manual_seed(123)

do=2

if do==1:
    embedded_sentence=do_embed()

    d=embedded_sentence.shape[1]
    print(d)
    d_q, d_k, d_v = 24, 24, 28
    W_query, W_key, W_value= define_weights(d,d_q, d_k, d_v)
    attention(embedded_sentence, W_query, W_key, W_value, d_q, d_k, d_v)

    d_in,d_out_kq,d_out_v=16,24,28
    torch.manual_seed(123)
    model = SelfAttention(d_in, d_out_kq, d_out_v)
    print(model(embedded_sentence))

if do==2:
    visualise()


