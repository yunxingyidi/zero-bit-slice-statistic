import torch
import torch.nn.functional as F
import pickle

# def split_string_by_length(string, length):
#     return [string[i:i+length] for i in range(0, len(string), length)]
#
# def count_zero_multiply(weigh_bit_slice, act_bit_slice, weight_bit_width, act_bit_width):
#     flag = []
#     mult = []
#     if weight_bit_width == 8 and act_bit_width == 8:
#         mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[3] == "00"))
#         mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[2] == "00"))
#         mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[3] == "00"))
#         mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[2] == "00"))
#         mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[1] == "00"))
#         mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[0] == "00"))
#         mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[1] == "00"))
#         mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[0] == "00"))
#         mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[3] == "00"))
#         mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[2] == "00"))
#         mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[3] == "00"))
#         mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[2] == "00"))
#         mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[1] == "00"))
#         mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[0] == "00"))
#         mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[1] == "00"))
#         mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[0] == "00"))
#         count = 0
#         for i in range(0, 16):
#             if mult[i] == True:
#                 count = count + 1
#             if ((i + 1) % 4) == 0:
#                 if count < 2:
#                     flag.append(0)
#                 else:
#                     flag.append(1)
#                 count = 0
#     wr = False
#     for f in flag:
#         if f == 0:
#             wr = True
#
#     return wr


# def statistic_one_multipy(weight_data, act_data, weight_bit_width, act_bit_width):
#     weight_count_zero_slice = 0
#     act_count_zero_slice = 0
#     weight_data_bin = format(weight_data & 0xff, f'0{weight_bit_width}b')
#     act_data_bin = format(act_data & 0xff, f'0{act_bit_width}b')
#     # print(weight_data_bin)
#     # print(act_data_bin)
#
#     weight_data_l = split_string_by_length(weight_data_bin, 2)
#     act_data_l = split_string_by_length(act_data_bin, 2)
#     c = 0
#     for i in range(len(weight_data_l) - 1, -1, -1):
#         if i < len(weight_data_l) - 1:
#             high = int(weight_data_l[i + 1][0])
#             c = c | high
#             next_value = int(weight_data_l[i], 2) + c
#             # print(next_value)
#             weight_data_l[i] = format(next_value, f'02b')[-2:]
#             c = int(format(next_value, f'02b')[:-1], 2)
#         if weight_data_l[i] == "00":
#             weight_count_zero_slice = weight_count_zero_slice + 1
#
#     c = 0
#     for i in range(len(act_data_l) - 1, -1, -1):
#         if i < len(act_data_l) - 1:
#             high = int(act_data_l[i + 1][0])
#             c = c | high
#             next_value = int(act_data_l[i], 2) + c
#             # print(next_value)
#             act_data_l[i] = format(next_value, f'02b')[-2:]
#             c = int(format(next_value, f'02b')[:-1], 2)
#         if act_data_l[i] == "00":
#             act_count_zero_slice = act_count_zero_slice + 1
#     count_zero_slice = {"weight" : weight_count_zero_slice, "activation" : act_count_zero_slice}
#     zero_multiply = count_zero_multiply(weight_data_l, act_data_l, 8, 8)
#
#     return zero_multiply


# import torch


# def analyse_linear(input_tensor, weight, bias=None):
#     # with open('zero_slice.txt', 'a') as file:
#     #     file.write("\n")
#     #     file.write("layer: Linear\n")
#     batch_size, in_features = input_tensor.shape
#     out_features, _ = weight.shape
#     linear_wr = 0
#     multiplications = batch_size * in_features * out_features
#
#     for b in range(batch_size):  # 遍历每个样本
#         for o in range(out_features):  # 遍历输出维度
#             for i in range(in_features):  # 遍历输入维度
#                 input_value = int(input_tensor[b, i])
#                 weight_value = int(weight[o, i])  # 逐元素乘法
#                 if not weight_value == 0:
#                     if statistic_one_multipy(weight_value, input_value, 8, 8):
#                         linear_wr = linear_wr + 1
#                     # with open('zero_slice.txt', 'a') as file:
#                     #     # file.write(str(weight_value))
#                     #     file.write(statistic_one_multipy(weight_value, input_value, 8, 8))
#                     #     file.write(",")
#                     # with open('linear_zero_slice.pkl', 'ab') as file:
#                     #     # file.write(str(weight_value))
#                     #     pickle.dump(statistic_one_multipy(weight_value, input_value, 8, 8), file)
#     print(linear_wr)
#     print(multiplications)
#     # return linear_wr / multiplications
#     # return multiplications
#
# def analyze_convolution(input_tensor, weight, stride=(1, 1), padding=(0, 0)):
#     N, C_in, H_in, W_in = input_tensor.shape
#     C_out, _, K_h, K_w = weight.shape
#
#     # 解析 padding 和 stride
#     pad_h, pad_w = padding
#     stride_h, stride_w = stride
#     conv_wr = 0
#     # 计算输出尺寸
#     H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
#     W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1
#     total_multiplications = N * C_out * H_out * W_out * C_in * K_h * K_w
#
#     # 修正 padding 4D 格式
#     input_padded = torch.nn.functional.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
#
#     # 遍历每个样本
#     for n in range(N):
#         for c_out in range(C_out):
#             for h in range(H_out):
#                 for w in range(W_out):
#                     # 计算输入窗口的起始位置
#                     h_start, w_start = h * stride_h, w * stride_w
#                     # 遍历每个输入通道和 kernel 位置
#                     for c_in in range(C_in):
#                         for i in range(K_h):
#                             for j in range(K_w):
#                                 # 确保索引在合法范围内
#                                 h_idx, w_idx = h_start + i, w_start + j
#                                 if 0 <= h_idx < H_in + 2 * pad_h and 0 <= w_idx < W_in + 2 * pad_w:
#                                     input_value = int(input_padded[n, c_in, h_idx, w_idx])
#                                     weight_value = int(weight[c_out, c_in, i, j])
#                                     if not weight_value == 0:
#                                         if statistic_one_multipy(weight_value, input_value, 8, 8):
#                                             conv_wr = conv_wr + 1
#                                             print("{}: {}".format(total_multiplications, conv_wr))
#                                         # with open('zero_slice.txt', 'a') as file:
#                                         #     # file.write(str(weight_value))
#                                         #     file.write(statistic_one_multipy(weight_value, input_value, 8, 8))
#                                         #     file.write(",")
#                                         # with open('conv_zero_slice.pkl', 'ab') as file:
#                                         #     # file.write(str(weight_value))
#                                         #     pickle.dump(statistic_one_multipy(weight_value, input_value, 8, 8), file)
#     print(conv_wr)
#     print(total_multiplications)

    # return conv_wr / total_multiplications
    # return total_multiplications



# input_tensor = torch.randint(-127, 128, (64, 64, 56, 56), dtype=torch.float32)
# weight = torch.randint(-127, 128, (64, 64, 1, 1), dtype=torch.float32)
#
#
# # 调用函数
# analyze_convolution(input_tensor, weight, stride=(1, 1), padding=(1, 1))
# import random
# print(statistic_one_multipy(random.randint(-127, 128), random.randint(-127, 128), 8, 8))


import torch

def count_zero_multiply_tensor(w_zero_mask, a_zero_mask):
    # 构造 16 个组合的 zero-mult 掩码，类似手动枚举
    mult_mask = []

    mult_mask.append(w_zero_mask[..., 3] | a_zero_mask[..., 3])
    mult_mask.append(w_zero_mask[..., 3] | a_zero_mask[..., 2])
    mult_mask.append(w_zero_mask[..., 2] | a_zero_mask[..., 3])
    mult_mask.append(w_zero_mask[..., 2] | a_zero_mask[..., 2])

    mult_mask.append(w_zero_mask[..., 3] | a_zero_mask[..., 1])
    mult_mask.append(w_zero_mask[..., 3] | a_zero_mask[..., 0])
    mult_mask.append(w_zero_mask[..., 2] | a_zero_mask[..., 1])
    mult_mask.append(w_zero_mask[..., 2] | a_zero_mask[..., 0])

    mult_mask.append(w_zero_mask[..., 1] | a_zero_mask[..., 3])
    mult_mask.append(w_zero_mask[..., 1] | a_zero_mask[..., 2])
    mult_mask.append(w_zero_mask[..., 0] | a_zero_mask[..., 3])
    mult_mask.append(w_zero_mask[..., 0] | a_zero_mask[..., 2])

    mult_mask.append(w_zero_mask[..., 1] | a_zero_mask[..., 1])
    mult_mask.append(w_zero_mask[..., 1] | a_zero_mask[..., 0])
    mult_mask.append(w_zero_mask[..., 0] | a_zero_mask[..., 1])
    mult_mask.append(w_zero_mask[..., 0] | a_zero_mask[..., 0])

    # 堆叠成 [..., 16]
    mult_mask_tensor = torch.stack(mult_mask, dim=-1)  # [..., 16]

    # 分成4组，每组4个，统计每组中True的个数
    mult_mask_tensor = mult_mask_tensor.view(*mult_mask_tensor.shape[:-1], 4, 4)  # [..., 4, 4]
    zero_counts = mult_mask_tensor.sum(dim=-1)  # [..., 4]

    # 如果某组少于2个 zero×zero，就记为 flag=0
    flags = (zero_counts >= 2).to(torch.int)  # [..., 4]

    # 如果有任何 flag==0，就返回 True
    wr_mask = (flags.sum(dim=-1) < 4)  # [...], True 表示需要处理
    return wr_mask


def split_2bit_fields(tensor, total_bits=8):
    """将张量拆分成2bit字段"""
    fields = []
    for i in range(0, total_bits, 2):
        field = (tensor >> i) & 0b11  # 取出第 i~i+1 bit
        fields.append(field)
    return torch.stack(fields[::-1], dim=-1)  # 逆序排列，最低位在后


def propagate_and_mask_zeros(bit_fields):
    """模拟进位传播和零位检查"""
    out = bit_fields.clone()
    c = torch.zeros_like(bit_fields[..., 0])  # 进位寄存器
    for i in range(3, -1, -1):
        if i < 3:
            high = (bit_fields[..., i + 1] >> 1) & 0b1
            c = c | high
            next_value = bit_fields[..., i] + c
            out[..., i] = next_value & 0b11
            c = (next_value >> 1) & 0b1
    zero_mask = (out == 0)  # 返回零位的掩码
    return zero_mask


def tensor_statistic_multipy(weight_tensor, act_tensor):
    """
    执行对权重和激活的统计操作，返回包含 zero multiply 的掩码。

    参数：
    - weight_tensor: 输入的权重张量
    - act_tensor: 输入的激活张量

    返回：
    - zero_mul_mask: 每次乘法是否满足 zero multiply 的掩码
    """
    # 将权重和激活拆分成2bit字段
    w_fields = split_2bit_fields(weight_tensor)
    a_fields = split_2bit_fields(act_tensor)

    # 执行进位传播和零位检查
    w_zero_mask = propagate_and_mask_zeros(w_fields)
    a_zero_mask = propagate_and_mask_zeros(a_fields)

    # 统计乘法中是否存在 zero × zero 的乘法
    # zero_mul_mask = (w_zero_mask & a_zero_mask).any(dim=-1)  # 每次卷积是否有零乘积
    zero_mul_mask = count_zero_multiply_tensor(w_zero_mask, a_zero_mask)
    return zero_mul_mask


# 示例：如何在卷积处理中使用
def analyze_convolution(input_tensor, weight, stride=(1, 1), padding=(0, 0)):
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, K_h, K_w = weight.shape

    # 解析 padding 和 stride
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    # 计算输出尺寸
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1
    total_multiplications = N * C_out * H_out * W_out * C_in * K_h * K_w

    # 用 PyTorch 的 unfold 展开 input 张量
    unfolded_input = torch.nn.functional.unfold(input_tensor, kernel_size=(K_h, K_w), stride=stride, padding=padding)
    unfolded_input = unfolded_input.transpose(1, 2)  # 转置维度以匹配相乘格式

    # 重塑权重
    weight_flat = weight.view(C_out, -1)

    # 执行 GPU 乘法操作，计算 zero slice 掩码
    weight_int = unfolded_input.int() & 0xff  # 保证权重是整数
    input_int = unfolded_input.int() & 0xff  # 保证输入是整数

    # 获取掩码，统计 zero multiply
    zero_mask = tensor_statistic_multipy(weight_int, input_int)
    ratio = zero_mask.flatten().float().mean().item()
    with open('conv_8x8.txt', 'a') as f:
        f.write(str(ratio))
        f.write(',')


def analyse_linear(input_tensor, weight, bias=None):
    batch_size, in_features = input_tensor.shape
    out_features, _ = weight.shape
    input_int = input_tensor.int() & 0xff
    weight_int = weight.int() & 0xff
    # 展开 input: [batch_size, in_features] → [batch_size, 1, in_features]
    # 展开 weight: [out_features, in_features] → [1, out_features, in_features]
    input_expand = input_int.unsqueeze(1)        # [B, 1, C]
    weight_expand = weight_int.unsqueeze(0)      # [1, O, C]

    # 广播得到所有乘法对：[B, O, C]
    input_broadcast = input_expand.expand(-1, out_features, -1)
    weight_broadcast = weight_expand.expand(batch_size, -1, -1)

    # 统计 zero multiply
    zero_mask = tensor_statistic_multipy(weight_broadcast, input_broadcast)  # [B, O]

    ratio = zero_mask.float().mean().item()
    with open('linear_8x8.txt', 'a') as f:
        f.write(str(ratio))
        f.write(',')



    # print(f"符合 zero multiply 条件的乘法数：{total_zero_like_mul}")
    # print(f"总乘法次数：{multiplications}")

# weight_tensor = torch.tensor([[1, 2], [3, 0]])  # 权重张量
# act_tensor = torch.tensor([[0, 3], [0, 0]])  # 激活张量
#
# # 调用tensor_statistic_multipy函数
# zero_mul_mask = tensor_statistic_multipy(weight_tensor, act_tensor)
#
# # 输出结果
# print("Zero Multiply Mask:")
# print(zero_mul_mask)
