
def inputprompt_to_outputprompt(model, tokenizer, prompt, split_str = None, max_new_token = 30):
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)

    # implementation
    '''
    with torch.no_grad():
        outputs = model.forward(
            **inputs,
            use_cache = True,
        )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    with torch.no_grad():
        for _ in range(max_new_token-1):
            outputs = model.forward(
                next_token,
                use_cache = True,
                past_key_values = past_key_values,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            tokens.append(next_token)

    final_outputs = torch.cat(tokens, dim=1)
    output_text = tokenizer.batch_decode(final_outputs, skip_special_tokens=True)
    '''

    # method
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_token,
        do_sample=True,
        temperature=0.8,
        top_p=1,
        min_p=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.batch_decode(outputs[:,-max_new_token:], skip_special_tokens=True)

    output_text = [
        t[1:max(t.rfind('.'), t.rfind('?')) + 1]
        if ('.' in t or '?' in t) else t[1:]
        for t in output_text
    ]
    if split_str != None:
        output_text = [t[:t.find(split_str)] if split_str in t else t for t in output_text]
    return output_text
