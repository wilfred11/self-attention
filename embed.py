import torch


def do_embed():
    inputs = ("According to the news, it it hard to say Melbourne is safe now")
    input_ids = {s: i for i, s in enumerate(sorted(inputs.replace(',', '').split()))}
    print(input_ids)
    input_tokens = torch.tensor([input_ids[s] for s in inputs.replace(',', '').split()])
    print(input_tokens)
    embed = torch.nn.Embedding(13, 16)
    embedded_sentence = embed(input_tokens).detach()
    print(embedded_sentence)
    print(embedded_sentence.shape)
    return embedded_sentence