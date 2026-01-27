import torch

def compute_logprob(logits, labels, mask):
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=1)
