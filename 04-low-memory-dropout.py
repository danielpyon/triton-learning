"""
Low-Memory Dropout
==================

In this tutorial, you will write a memory-efficient implementation of dropout whose state
will be composed of a single int32 seed. This differs from more traditional implementations of dropout,
whose state is generally composed of a bit mask tensor of the same shape as the input.

In doing so, you will learn about:

* The limitations of naive implementations of Dropout with PyTorch.

* Parallel pseudo-random number generation in Triton.

"""

# %%
# Baseline
# --------
#
# The *dropout* operator was first introduced in [SRIVASTAVA2014]_ as a way to improve the performance
# of deep neural networks in low-data regime (i.e. regularization).
#
# It takes a vector as input and produces a vector of the same shape as output. Each scalar in the
# output has a probability :math:`p` of being changed to zero and otherwise it is copied from the input.
# This forces the network to perform well even when only :math:`1 - p` scalars from the input are available.
#
# At evaluation time we want to use the full power of the network so we set :math:`p=0`. Naively this would
# increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease
# in the output softmax temperature). To prevent this we multiply the output by :math:`\frac{1}{1 - p}`, which
# keeps the norm consistent regardless of the dropout probability.
#
# Let's first take a look at the baseline implementation.

import tabulate
import torch
import triton_viz
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
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# %%
# Seeded dropout
# --------------
#
# The above implementation of dropout works fine, but it can be a bit awkward to deal with. Firstly
# we need to store the dropout mask for backpropagation. Secondly, dropout state management can get
# very tricky when using recompute/checkpointing (e.g. see all the notes about `preserve_rng_state` in
# https://pytorch.org/docs/1.9.0/checkpoint.html). In this tutorial we'll describe an alternative implementation
# that (1) has a smaller memory footprint; (2) requires less data movement; and (3) simplifies the management
# of persisting randomness across multiple invocations of the kernel.
#
# Pseudo-random number generation in Triton is simple! In this tutorial we will use the
# :code:`triton.language.rand` function which generates a block of uniformly distributed :code:`float32`
# values in [0, 1), given a seed and a block of :code:`int32` offsets. But if you need it, Triton also provides
# other :ref:`random number generation strategies<Random Number Generation>`.
#
# .. note::
#    Triton's implementation of PRNG is based on the Philox algorithm (described on [SALMON2011]_).
#
# Let's put it all together.

@triton_viz.trace
@triton.jit
def _seeded_dropout(
    x_ptr,
    x_row_stride,
    output_ptr,
    n_cols,
    p,
    seeds,
    BLOCK_SIZE: tl.constexpr,
):
    # parallelize over rows of input matrix
    row = tl.program_id(axis=0)
    seed = seeds[row]

    # BLOCK_SIZE may be greater than n_cols
    offsets = x_row_stride * row + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    # print(offsets * mask)

    # load x
    x = tl.load(x_ptr + offsets, mask=mask)

    # dropout
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seeds):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    assert x.is_contiguous()

    # note: grid is now number of rows, not numel / blocksize
    _seeded_dropout[(n_rows,)](
        x,
        x.stride(0),
        output,
        n_cols,
        p,
        seeds,
        BLOCK_SIZE=triton.next_power_of_2(n_cols)
    )

    return output

if __name__ == '__main__':
    device = 'cpu'

    '''
    # Input tensor
    x = torch.randn(size=(10, ), device=device)
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10, ), device=device) > p).to(torch.int32)

    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))
    '''

    x = torch.randn(size=(3,5), device=device)
    seeds = [123, 123, 512]
    output = seeded_dropout(x, p=0.1, seeds=seeds)

    print(output)

# %%
# Et Voilà! We have a triton kernel that applies the same dropout mask provided the seed is the same!
# If you'd like explore further applications of pseudorandomness in GPU programming, we encourage you
# to explore the `triton/language/random` folder!

# %%
# Exercises
# ---------
#
# 1. Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
# 2. Add support for striding.
# 3. (challenge) Implement a kernel for sparse Johnson-Lindenstrauss transform which generates the projection matrix on the fly each time using a seed.

# %%
# References
# ----------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
