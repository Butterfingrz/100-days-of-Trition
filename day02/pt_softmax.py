import torch

def softmax(x):
    x_exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

#! 一共 读取：5MN + 2M   写入：3MN + 2M
def softmax_detail(x):
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

if __name__ == '__main__':
    input_tensor = torch.randn(1000, 512)
    output_custom = softmax_detail(input_tensor)
    print(output_custom)