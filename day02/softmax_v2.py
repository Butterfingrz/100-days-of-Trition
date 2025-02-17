import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_2_rows(
    output_ptr, # 输出张量的指针
    input_ptr, # 输入张量的指针
    input_row_stride, # 输入张量的行步长
    output_row_stride, # 输出张量的行步长
    n_rows, # 行数
    n_cols, # 列数
    BLOCK_SIZE: tl.constexpr, # 块大小
):
    row_len = 2
    row_start = tl.program_id(0) * row_len
    if row_start >= n_rows:
        return
    for row_idx in range(row_start, row_start + row_len, 1): #* 一个block处理两行，因此要写一个循环
        #* 定位一行的起点
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0 , BLOCK_SIZE)
        #* 表示某行所有元素的下标
        input_ptrs = row_start_ptr + col_offset
        mask = col_offset < n_cols
        #* 加载一行的所有元素
        row = tl.load(input_ptrs, mask=mask, other = -float('inf'))

        #* 计算
        row_minus_max = row - tl.max(row, axis = 0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis = 0)
        softmax_output = numerator / denominator

        #* 写回定位一行的起点
        out_row_start_ptr = output_ptr + row_idx * output_row_stride
        out_ptrs = out_row_start_ptr + col_offset

        #* 写回
        tl.store(out_ptrs, softmax_output, mask=mask)


def softmax_2_rows_v2(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.ndim == 2

    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    # BLOCK_SIZE = 256
    # num_stages = 4
    #根据输入大小配置不同的参数
    if n_cols == 512:
        BLOCK_SIZE = 512
        num_stages = 3
    elif n_cols == 1024:
        BLOCK_SIZE = 1024
        num_stages = 4
    elif n_cols == 2048:
        BLOCK_SIZE = 512
        num_stages = 5
    else:
        # 默认配置
        BLOCK_SIZE = 2048
        num_stages = 3

    # 调用kernel
    grid = lambda meta : (triton.cdiv(n_rows,2),)
    softmax_kernel_2_rows[grid](output,
                                x,
                                x.stride(0),
                                output.stride(0),
                                n_rows,
                                n_cols,
                                BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages)

# #* 验证
# expected_output = torch.softmax(x, dim=1)
# print("Triton Softmax 和 PyTorch Softmax 是否接近:", torch.allclose(output, expected_output, atol=1e-6))