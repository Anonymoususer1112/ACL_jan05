import torch
from contextlib import contextmanager

def get_decoder_blocks(model):
    # Phi / Llama style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # NeoX style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Cannot find transformer blocks on this model.")

def hs_idx_to_block_idx(model, hs_idx: int) -> int:
    # hidden_states[0] = embeddings, hidden_states[1] = after block 0
    blocks = get_decoder_blocks(model)
    block_idx = hs_idx - 1
    if not (0 <= block_idx < len(blocks)):
        raise ValueError(f"hs_idx={hs_idx} -> block_idx={block_idx} out of range (num_blocks={len(blocks)})")
    return block_idx

@contextmanager
def dialect_steering_hook(model, v_unit: torch.Tensor, hs_idx: int, beta: float):
    """
    Applies: h <- h + beta * v_unit on the output hidden state of transformer block (hs_idx-1).
    """
    blocks = get_decoder_blocks(model)
    block_idx = hs_idx_to_block_idx(model, hs_idx)

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    v = v_unit.to(device=device, dtype=dtype)

    def hook_fn(module, inputs, output):
        if beta == 0.0:
            return output

        # output could be Tensor or tuple(list) with first element Tensor
        if torch.is_tensor(output):
            return output + beta * v

        if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            out0 = output[0] + beta * v
            if isinstance(output, tuple):
                return (out0,) + tuple(output[1:])
            new_out = list(output)
            new_out[0] = out0
            return new_out

        return output

    handle = blocks[block_idx].register_forward_hook(hook_fn)
    try:
        yield {"hs_idx": hs_idx, "block_idx": block_idx, "beta": beta}
    finally:
        handle.remove()