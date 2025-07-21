import torch
import triton
import triton.language as tl


@torch.compile
def compiled_softmax(x):
    # Step 1: 计算最大值（按行）
    max_val = torch.max(x,  dim=-1, keepdim=True).values  #* 读取 M*N，写入 M*1 
    # Step 2: 减去最大值（防止数值溢出）
    x_sub = x - max_val                                   #* 读取 M*N + M*1，写入 M*N 
    # Step 3: 指数运算 
    x_exp = torch.exp(x_sub)                               #* 读取 M*N，写入 M*N 
    # Step 4: 计算分母（按行的和）
    sum_exp = torch.sum(x_exp,  dim=-1, keepdim=True)      #* 读取 M*N，写入 M*1 
    # Step 5: 归一化 
    output = x_exp / sum_exp                              #* 读取 M*N + M*1，写入 M*N 
    return output 

if __name__ == "__main__":
    print("测试小矩阵...")
    x_small = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    y_small = compiled_softmax(x_small)
    print(y_small)