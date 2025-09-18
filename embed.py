import torch


def do_embed():
    #inputs = ("According to the news, it it hard to say Melbourne is safe now")
    sentence="Write a poem about a man fishing on a river bank"

    input_ids = {s: i for i, s in enumerate(sorted(sentence.split()))}
    print(input_ids)
    input_tokens = torch.tensor([input_ids[s] for s in sentence.replace(',', '').split()])
    print(input_tokens)
    embed = torch.nn.Embedding(11, 16)
    embedded_sentence = embed(input_tokens).detach()
    print('embedded sentence')
    print(embedded_sentence)
    print('embedded sentence shape')
    print(embedded_sentence.shape)
    return embedded_sentence