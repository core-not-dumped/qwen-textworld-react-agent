import torch

def token_logprobs(logits, labels):
    """
    logits: [B, T, V]
    labels: [B, T]
    """
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)

def sequence_logp(logits, labels, mask):
    token_lp = token_logprobs(logits, labels)
    return (token_lp * mask).sum(dim=-1)


