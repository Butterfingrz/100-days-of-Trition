import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )


@triton.jit
def matmul_kernal_v1(
    input_ptr,
    weight_ptr,
    output_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    Z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(input_ptr + offset_m * K + x_k, mask=(offset_m < M) & (x_k < K), other=0.0)
        x = x.to(tl.float16)

        y_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        y = tl.load(weight_ptr + y_k * N + offset_n, mask=(offset_n < N) & (y_k < K), other=0.0)
        y = y.to(tl.float16)

        Z += tl.dot(x, y)

    output_offset = offset_m * N + offset_n
    output_mask = (offset_m < M) & (offset_n < N)

    tl.store(output_ptr + output_offset, Z, mask=output_mask)

@torch.no_grad()
def matmul_v1(
     x : torch.Tensor,
     weight : torch.Tensor,
):
    
    out_shape_0 = x.shape[:-1]  #* 是一个元组，包含了 x 除了最后一个维度之外的所有维度的大小，即 (d1, d2, ..., dn-1)。
    x = x.view(-1, x.shape[-1]) #* (d1, d2, ..., dn-1, dn) ---> (d1 * d2 * ... * dn-1, dn)。 
                                    #* 也就是说它把除了最后一个维度外的其他维度合并成一个维度

    M, K = x.shape
    N = weight.shape[1]

    assert x.ndim == 2 and weight.ndim == 2
    assert x.shape[1] == weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    # num_stages: tl.constexpr = 3,     # 可以尝试减小到2或1
    # num_warps: tl.constexpr = 4       # 可以尝试4或8
    grid = ((triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1))

    matmul_kernal_v1[grid](
        x,
        weight,
        out,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out.view(out_shape_0 + (N,)) #* *out_shape_0: 这是一个解包操作。
                                        #* (d1 * d2 * ... * dn-1, dn)--->(d1, d2, ..., dn-1, N)




   
    


