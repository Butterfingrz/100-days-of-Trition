import triton
import triton.language as tl
import torch
from triton.runtime import driver

# # #* 定义自动调优配置
# @triton.autotune(
#     configs=[
#         triton.Config({'num_stages': 1, 'num_warps': 1}, num_warps=1),
#         triton.Config({'num_stages': 1, 'num_warps': 2}, num_warps=2),
#         triton.Config({'num_stages': 1, 'num_warps': 4}, num_warps=4),
#         triton.Config({'num_stages': 1, 'num_warps': 8}, num_warps=8),
#         triton.Config({'num_stages': 2, 'num_warps': 1}, num_warps=1),
#         triton.Config({'num_stages': 2, 'num_warps': 2}, num_warps=2),
#         triton.Config({'num_stages': 2, 'num_warps': 4}, num_warps=4),
#         triton.Config({'num_stages': 2, 'num_warps': 8}, num_warps=8),
#         triton.Config({'num_stages': 4, 'num_warps': 1}, num_warps=1),
#         triton.Config({'num_stages': 4, 'num_warps': 2}, num_warps=2),
#         triton.Config({'num_stages': 4, 'num_warps': 4}, num_warps=4),
#         triton.Config({'num_stages': 4, 'num_warps': 8}, num_warps=8),
#     ],
#     key=['n_rows', 'n_cols'],  #* 根据输入形状选择最佳配置
# )
@triton.jit
def softmax_kernel_v2(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 1,
    num_warps: tl.constexpr = 4, 
):
    #* （1）获取当前线程的行索引
    row_start = tl.program_id(0)   #! 当前的program id
    row_step = tl.num_programs(0)  #! 总program数量

    #*  (2) 使用已有线程块跨步循环遍历所有行
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        #* （3）获取当前线程索引
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offset

        #* （4）加载当前线程的数据
        row = tl.load(input_ptrs, mask=col_offset < n_cols, other=-float('inf'))

        #* （5）计算当前线程的softmax输出
        row_minus_max = row - tl.max(row)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator)
        softmax_output = numerator / denominator

        #* （6）写回当前线程的softmax输出
        output_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_ptr + col_offset
        tl.store(output_ptrs, softmax_output, mask=col_offset < n_cols)

# DEVICE = triton.runtime.driver.active.get_active_torch_device()

# properties = driver.active.utils.get_device_properties(DEVICE.index)
# NUM_SM = properties["multiprocessor_count"]
# NUM_REGS = properties["max_num_regs"]
# SIZE_SMEM = properties["max_shared_mem"]
# WARP_SIZE = properties["warpSize"]
# target = triton.runtime.driver.active.get_current_target()

def softmax_v2(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    grid = (n_rows, )
    #* 调用自动调优的kernel，不需要手动指定num_stages和num_warps
    softmax_kernel_v2[grid](
        x,
        y,
        n_rows,
        n_cols,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE,
    )
    return y

if __name__ == "__main__":
    print("测试小矩阵...")
    x_small = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    y_small = softmax_v2(x_small)
    assert torch.allclose(y_small, torch.softmax(x_small,dtype=torch.float16,dim=1), atol=1e-3, rtol=1e-3)
    print("小矩阵测试通过！")
    
    print("测试中等矩阵...")
    x_medium = torch.randn(8192, 8192, dtype=torch.float16, device="cuda")
    y_medium = softmax_v2(x_medium)
    assert torch.allclose(y_medium, torch.softmax(x_medium, dtype=torch.float16, dim=1), atol=1e-3, rtol=1e-3)
    print("中等矩阵测试通过！")
    
    print("测试大矩阵...")
    x_large = torch.randn(2048, 151936, dtype=torch.float16, device="cuda")
    y_large = softmax_v2(x_large)
    assert torch.allclose(y_large, torch.softmax(x_large, dtype=torch.float16, dim=1), atol=1e-3, rtol=1e-3)
    print("大矩阵测试通过！")    