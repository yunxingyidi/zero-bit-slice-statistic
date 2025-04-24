import torch
import torch.nn.functional as F
import pickle
import math

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
    """将张量拆分成2bit字段，低位在前"""
    fields = []
    for i in range(0, total_bits, 2):
        field = (tensor >> i) & 0b11  # 取出第 i~i+1 bit
        fields.append(field)
    return torch.stack(fields, dim=-1)


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

# def propagate_and_zeros(bit_fields):
#     out = bit_fields.clone()
#     c = torch.zeros_like(bit_fields[..., 0])
#     c_i2 = None
#     out[..., 0] = bit_fields[..., 0]
#     for i in range(0, 4):
#         if i < 3 and i > 0:
#             high = (out[..., i - 1] >> 1) & 0b1
#             c = c | high
#             next_value = bit_fields[..., i] + c
#             out[..., i] = next_value & 0b11
#             c = (next_value >> 2) & 0b1
#             if i == 2:
#                 c_i2 = c | ((out[..., i] >> 1) & 0b1)
#         else:
#             out[..., i] = bit_fields[..., i]
#     out = torch.where(out < 2, out, out - 4)
#     return out, c_i2
def propagate_and_zeros(bit_fields):
    out = bit_fields.clone()
    c = torch.zeros_like(bit_fields[..., 0])
    c_i2 = None
    for i in range(0, 4):
        if i > 0:
            high = (out[..., i - 1] >> 1) & 0b1
            c = c | high
            next_value = bit_fields[..., i] + c
            out[..., i] = next_value & 0b11
            if i == 3:
                c_i2 = (bit_fields[..., i] < 2) & (c == 1)
                out[..., i][c_i2] = bit_fields[..., i][c_i2]
            c = (next_value >> 2) & 0b1
            # if i == 2:
            #     c_i2 = c | ((out[..., i] >> 1) & 0b1)
        else:
            out[..., i] = bit_fields[..., i]
    out = torch.where(out < 2, out, out - 4)
    return out, c_i2.int()


def tensor_statistic_multipy(weight_tensor, act_tensor):
    # 将权重和激活拆分成2bit字段
    w_fields = split_2bit_fields(weight_tensor)
    a_fields = split_2bit_fields(act_tensor)

    # 执行进位传播和零位检查
    w_zero_mask = propagate_and_mask_zeros(w_fields)
    a_zero_mask = propagate_and_mask_zeros(a_fields)

    # 统计乘法中是否存在 zero × zero 的乘法
    zero_mul_mask = count_zero_multiply_tensor(w_zero_mask, a_zero_mask)
    return zero_mul_mask

#
# def simulate_2bit_matmul(w_fields, a_fields, w, a, w_carry, a_carry):
#     sample_out = torch.matmul(a.float(), w.float())  # shape: [B, O, L]
#     # 初始化结果张量，使用sample_out的形状（注意 transpose）
#     result = torch.zeros_like(sample_out, dtype=torch.int32)  # [B, O, L]
#     # [B, S, O, K], [B, S, K, L]
#     shift_pairs = [
#         (0, 0), (0, 1), (1, 0), (1, 1),
#         (0, 2), (0, 3), (1, 2), (1, 3),
#         (2, 0), (2, 1), (3, 0), (3, 1),
#         (2, 2), (2, 3), (3, 2), (3, 3),
#     ]
#
#     for i, j in shift_pairs:
#         # w_ij = w_fields[:, i, :, :]  # [B, O, K]
#         # a_ij = a_fields[:, j, :, :]  # [B, K, L]
#         w_ij = w_fields[i, ...]  # [B, O, K]
#         a_ij = a_fields[j, ...]  # [B, K, L]
#         p_ij = torch.matmul(a_ij.float(), w_ij.float())  # [B, O, L]
#         result += p_ij.to(torch.int32) << ((i + j) * 2)
#     # result = result.squeeze(0)  # [1, L]
#     # expand 以便做矩阵乘法
#     a_shifted = a.to(torch.int32) << 6  # [B, L, K]
#     w_shifted = w.to(torch.int32) << 6  # [B, O, K]
#
#     extra1 = torch.matmul(a_shifted.float(), w_carry.float()).to(torch.int32)  # [B, O, L]
#     extra2 = torch.matmul(a_carry.float(), w_shifted.float()).to(torch.int32)  # [B, O, L]
#
#     carry_and = (torch.matmul(a_carry.float(), w_carry.float()).to(torch.int32) << 12)  # [B, O, L]
#     result += extra1 + extra2 - carry_and
#
#     return result  # [B, O, L]

def simulate_2bit_matmul(w_fields, a_fields, w, a, w_carry, a_carry):
    sample_out = torch.matmul(a.float(), w.float())  # shape: [B, O, L]
    # 初始化结果张量，使用sample_out的形状（注意 transpose）
    result = torch.zeros_like(sample_out, dtype=torch.int32)  # [B, O, L]
    shift_pairs = [
            (3, 0), (0, 3), (0, 1), (1, 0),
            (3, 1), (1, 3), (0, 2), (2, 0),
            (3, 2), (2, 3), (1, 2), (2, 1),
            (3, 3), (2, 2), (1, 1), (0, 0)
    ]

    # 每组 4 个
    for group_idx in range(0, len(shift_pairs), 4):
        group = shift_pairs[group_idx:group_idx + 4]
        count_tensor = None  # 当前组的计数器
        for i, j in group:
            w_ij = w_fields[i, ...]  # [B, O, K]
            a_ij = a_fields[j, ...]  # [B, K, L]
            a_expanded = a_ij.unsqueeze(-1).float()  # [B, K, L, 1]
            w_expanded = w_ij.unsqueeze(-3).float()  # [B, O, 1, K]
            mul_tensor = a_expanded * w_expanded  # [B, O, K, L]
            # print((mul_tensor != 0).int())
            # 初始化或累加 count_tensor
            if count_tensor is None:
                count_tensor = (mul_tensor != 0).int()
            else:
                count_tensor += (mul_tensor != 0).int()

            # 抑制掉 count > 2 的位置
            mul_tensor[count_tensor > 2] = 0
            p_ij = mul_tensor.transpose(-2, -1).sum(dim=-1)  # [B, O, L]
            result += p_ij.to(torch.int32) << ((i + j) * 2)

        # ratio = (count_tensor > 2).sum().item() / count_tensor.numel()
        # print(f"Group {group_idx // 4} - >2 占比: {ratio:.4%}")
    # result = result.squeeze(0)  # [1, L]
    # expand 以便做矩阵乘法
    a_shifted = a.to(torch.int32) << 6              # [B, L, K]
    w_shifted = w.to(torch.int32) << 6              # [B, O, K]

    extra1 = torch.matmul(a_shifted.float(), w_carry.float()).to(torch.int32)  # [B, O, L]
    extra2 = torch.matmul(a_carry.float(), w_shifted.float()).to(torch.int32) # [B, O, L]

    carry_and = (torch.matmul(a_carry.float(), w_carry.float()).to(torch.int32) << 12) # [B, O, L]
    result += extra1 + extra2 - carry_and

    return result  # [B, O, L]

def simulate_2bit_matmul_dw(w_fields, a_fields, w, a, w_carry, a_carry):
    sample_out = torch.matmul(a.float(), w.float())  # shape: [B, O, L]
    # 初始化结果张量，使用sample_out的形状（注意 transpose）
    result = torch.zeros_like(sample_out, dtype=torch.int32)  # [B, O, L]

    shift_pairs = [
        (3, 0), (0, 3), (0, 1), (1, 0),
        (3, 1), (1, 3), (0, 2), (2, 0),
        (3, 2), (2, 3), (1, 2), (2, 1),
        (3, 3), (2, 2), (1, 1), (0, 0)
    ]

    for group_idx in range(0, len(shift_pairs), 4):
        group = shift_pairs[group_idx:group_idx + 4]
        count_tensor = None
        for i, j in group:
            w_ij = w_fields[i]  # [B, C, K]
            a_ij = a_fields[j]  # [B, C, K, L]

            a_exp = a_ij.unsqueeze(-1).float()  # [B, C, K, L]
            w_exp = w_ij.unsqueeze(-3).float()  # [B, C, K, 1]

            mul = a_exp * w_exp  # [B, C, K, L]

            if count_tensor is None:
                count_tensor = (mul != 0).int()
            else:
                count_tensor += (mul != 0).int()

            mul[count_tensor > 2] = 0
            p_ij = mul.sum(dim=3)  # sum over K → [B, C, L]
            result += p_ij.to(torch.int32) << ((i + j) * 2)

    # >>> carry 补偿
    a_shifted = a.to(torch.int32) << 6       # [B, C, K, L]
    w_shifted = w.to(torch.int32) << 6       # [B, C, K]

    extra1 = torch.matmul(a_shifted.float(), w_carry.float()).to(torch.int32)  # [B, O, L]
    extra2 = torch.matmul(a_carry.float(), w_shifted.float()).to(torch.int32)  # [B, O, L]

    carry_and = (torch.matmul(a_carry.float(), w_carry.float()).to(torch.int32) << 12)  # [B, O, L]

    result += extra1 + extra2 - carry_and
    return result  # [B, C, L]


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

def compute_convolution(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # 计算输出空间大小
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

    # unfold 展开为 im2col 格式
    input_unfold = F.unfold(input_tensor.float(), kernel_size=(K_h, K_w), stride=(stride_h, stride_w),
                            padding=(pad_h, pad_w))  # [N, K, L]
    input_unfold = input_unfold.transpose(1, 2)  # [N, L, K]

    # 权重 reshape 并准备 bit-slice
    weight_flat = weight.reshape(C_out, -1)  # [C_out, K]
    weight_exp = weight_flat.unsqueeze(0).transpose(1, 2)  # [1, C_out, K]
    weight_int = weight_exp.int() & 0xff  # 模拟 8bit 整数

    weight_fields = split_2bit_fields(weight_int, total_bits=8)  # [1, C_out, K, 4]
    weight_fields, w_carry = propagate_and_zeros(weight_fields)  # [1, C_out, K, 4], [1, C_out, K]

    # 输入也进行 bit-slice 拆分
    input_exp = input_unfold  # [N, L, 1, K]
    input_int = input_exp.int() & 0xff
    input_fields = split_2bit_fields(input_int, total_bits=8)  # [N, L, 1, K, 4]
    input_fields, a_carry = propagate_and_zeros(input_fields)  # [...], [N, L, K]
    # output = torch.matmul(input_exp.float(), weight_exp.float())
    weight_fields = weight_fields.permute(3, 0, 1, 2)  # [B, 4, O, K]
    input_fields = input_fields.permute(3, 0, 1, 2)  # [B, 4, K, L]


    # 模拟 bit-slice 的乘法
    sim_result = simulate_2bit_matmul(
        w_fields=weight_fields,  # [1, C_out, K, 4]
        a_fields=input_fields,  # [N, 1, K, 4]
        w=weight_exp.to(torch.int32),  # [1, C_out, K]
        a=input_exp.to(torch.int32),  # [N, K]
        w_carry=w_carry,  # [1, C_out, K]
        a_carry=a_carry  # [N, K]
    )  # → [N, C_out, K]

    # 汇总乘法结果（按K维度求和）
    output = sim_result  # [N, C_out]
    # output shape 还原为 [N, C_out, H_out, W_out]
    output = output.view(N, H_out * W_out, C_out).permute(0, 2, 1).contiguous()
    output = output.view(N, C_out, H_out, W_out)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


def compute_linear(input_tensor, weight, bias=None):
    # Step 1: 拆分为2bit字段，返回 shape [B, C_in, 4]
    input_q = input_tensor.int() & 0xFF
    weight_q = weight.int() & 0xFF

    input_fields = split_2bit_fields(input_q, total_bits=8)  # [B, C_in, 4]
    weight_fields = split_2bit_fields(weight_q, total_bits=8)  # [C_out, C_in, 4]

    # Step 2: 传播符号位（带carry），输出同样shape的signed字段
    input_fields, a_carry = propagate_and_zeros(input_fields)  # [B, C_in, 4]
    weight_fields, w_carry = propagate_and_zeros(weight_fields)  # [C_out, C_in, 4]

    weight_fields = weight_fields.permute(2, 0, 1)  # [B, 4, O, K]
    input_fields = input_fields.permute(2, 0, 1)  # [B, 4, K, L]

    output = simulate_2bit_matmul(
        w_fields=weight_fields,  # [1, C_in, C_out, 4]
        a_fields=input_fields,  # [B, C_in, 4]
        w=weight,
        a=input_tensor,
        w_carry=w_carry,
        a_carry=a_carry
    )  # [B, C_out]
    if bias is not None:
        output = output + bias.view(1, -1)

    return output

def compute_dw_convolution(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    N, C_in, H_in, W_in = input_tensor.shape
    _, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # 计算输出空间大小
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

    # unfold 展开为 im2col 格式
    input_unfold = F.unfold(input_tensor.float(), kernel_size=(K_h, K_w), stride=(stride_h, stride_w),
                            padding=(pad_h, pad_w))  # [N, C_in * K_h*K_w, L]
    input_unfold = input_unfold.view(N, C_in, K_h * K_w, -1)  # [N, C_in, K, L]
    input_unfold = input_unfold.permute(0, 1, 3, 2)  # [N, C_in, L, K]

    # reshape 权重：DW卷积下，每个输入通道有自己的卷积核
    weight_flat = weight.view(C_in, -1)  # [C_in, K]
    weight_exp = weight_flat.unsqueeze(0).unsqueeze(-1)  # [1, C_in, K]

    # int8 转换 + bit-slice 拆分
    weight_int = weight_exp.int() & 0xff  # [1, C_in, K]
    weight_fields = split_2bit_fields(weight_int, total_bits=8)  # [1, C_in, K, 4]
    weight_fields, w_carry = propagate_and_zeros(weight_fields)  # [1, C_in, K, 4], [1, C_in, K]

    input_int = input_unfold.int() & 0xff  # [N, C_in, L, K]
    input_fields = split_2bit_fields(input_int, total_bits=8)  # [N, C_in, L, K, 4]
    input_fields, a_carry = propagate_and_zeros(input_fields)  # [N, C_in, L, K, 4], [N, C_in, L, K]

    # 调整维度顺序：将 bit-slice 提到最前面，方便逐slice乘法
    weight_fields = weight_fields.permute(4, 0, 1, 2, 3)        # [4, 1, C_in, K]
    input_fields = input_fields.permute(4, 0, 1, 2, 3)       # [4, N, C_in, L, K]
    # 模拟 bit-slice DW 乘法：按通道分别进行（没有跨通道）
    sim_result = simulate_2bit_matmul_dw(
        w_fields=weight_fields,  # [B, 1, C_in, K]
        a_fields=input_fields,   # [B, N, C_in, L, K]
        w=weight_exp.to(torch.int32),     # [1, C_in, K]
        a=input_unfold.to(torch.int32),   # [N, C_in, L, K]
        w_carry=w_carry,  # [1, C_in, K]
        a_carry=a_carry   # [N, C_in, L, K]
    )  # [N, C_in, L]

    # reshape 回 feature map 格式
    output = sim_result.view(N, C_in, H_out, W_out)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output
