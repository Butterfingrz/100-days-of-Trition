import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    #* （1）获取当前线程的行索引
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offset = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offset

    #* （2）加载当前线程的数据
    row = tl.load(input_ptrs, mask=col_offset < n_cols)

    #* （3）计算当前线程的softmax输出
    row_minus_max = row - tl.max(row)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator)
    softmax_output = numerator / denominator

    #* （4）写回当前线程计算的softmax输出
    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offset
    tl.store(output_ptrs, softmax_output, mask=col_offset < n_cols)


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 1
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](
        x,
        y,
        n_rows=n_rows,
        n_cols=n_cols,
        input_row_stride=x.stride(0),
        output_row_stride=y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


if __name__ == "__main__":
    print("测试小矩阵...")
    x_small = torch.randn(2048, 1024, dtype=torch.float16, device="cuda")
    y_small = softmax(x_small)
    #assert torch.allclose(y_small, torch.softmax(x_small,dtype=torch.float16,dim=1), atol=1e-3, rtol=1e-3)

    
