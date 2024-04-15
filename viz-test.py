import torch
import triton_viz
import argparse
import tabulate

import triton
import triton.language as tl

@triton_viz.trace
@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # input
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # compute
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # output
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(device, x, x_keep, p):
    output = torch.empty_like(x, device=device)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = args.device

    triton_viz.sample((0,))

    # Input tensor
    x = torch.randn(size=(10, ), device=device)
    # Dropout mask
    p = 0.5

    x_keep = (torch.rand(size=(10, )) > p).to(torch.int32)
    
    output = dropout(device, x, x_keep=x_keep, p=p)
    output2 = dropout(device, x, x_keep=x_keep, p=p)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output2.tolist(),
        ]))

    # triton_viz.launch()
