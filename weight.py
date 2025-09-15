import torch
import torch.nn.functional as F

def define_weights(d, d_q, d_k, d_v):
    torch.manual_seed(123)


    W_query = torch.nn.Parameter(torch.rand(d_q, d))
    W_key = torch.nn.Parameter(torch.rand(d_k, d))
    W_value = torch.nn.Parameter(torch.rand(d_v, d))

    print(W_query.shape)
    print(W_key.shape)
    print(W_value.shape)
    print(W_value)
    return W_query, W_key, W_value

def attention(embedded_sentence, W_query,W_key, W_value, d_q, d_k, d_v):
    x_2 = embedded_sentence[1]
    query_2 = W_query.matmul(x_2)
    key_2 = W_key.matmul(x_2)
    value_2 = W_value.matmul(x_2)
    print(value_2.shape)
    print(value_2)

    keys = W_key.matmul(embedded_sentence.T).T
    values = W_value.matmul(embedded_sentence.T).T
    print(keys.shape)
    print(values.shape)

    w_2_4 = query_2.dot(keys[4])
    print(w_2_4)

    w_2 = query_2.matmul(keys.T)
    print(w_2)

    attention_weights_2 = F.softmax(w_2 / d_k ** 0.5, dim=0)
    print(attention_weights_2)
    print(attention_weights_2.shape)
    context_vector_2 = attention_weights_2.matmul(values)
    print(context_vector_2)
    print(context_vector_2.shape)