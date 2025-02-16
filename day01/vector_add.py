import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, # 第一个向量的指针
    y_ptr, # 第二个向量的指针
    output_ptr, # 输出向量的指针
    n_elements, # 向量的长度
    BLOCK_SIZE: tl.constexpr, #* 通过将这个值设为常量，编译器可以生成更高效的代码，因为知道这个值不会改变
):
   # 获取当前程序的全局索引
   pid = tl.program_id(0)
   # 计算当前程序的块起始索引
   block_start = pid * BLOCK_SIZE
   # 计算当前程序的块结束索引
   offsets = block_start + tl.arange(0, BLOCK_SIZE)
   # 计算当前程序的掩码
   mask = offsets < n_elements
   # 加载当前程序的向量
   x = tl.load(x_ptr + offsets, mask=mask)
   y = tl.load(y_ptr + offsets, mask=mask)
   output = x + y
   tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor):
   # 确保输入张量在 GPU 上并且是连续的
   assert x.is_cuda and y.is_cuda
   assert x.is_contiguous() and y.is_contiguous()
   assert x.shape == y.shape

   output = torch.empty_like(x)
   n_elements = x.numel()
   BLOCK_SIZE = 1024
   # 计算 grid 的大小
   grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
   vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
   return output


if __name__ == "__main__":
   x = torch.randn(1000000, device="cuda")
   y = torch.randn(1000000, device="cuda")
   output = vector_add(x , y)

   assert torch.allclose(output, x + y)
   print("Kernel executed successfully!")


   