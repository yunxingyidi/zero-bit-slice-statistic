import torch
import torch.nn.functional as F

def split_string_by_length(string, length):
    return [string[i:i+length] for i in range(0, len(string), length)]

def count_zero_multiply(weigh_bit_slice, act_bit_slice, weight_bit_width, act_bit_width):
    flag = []
    mult = []
    if weight_bit_width == 8 and act_bit_width == 8:
        mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[3] == "00"))
        mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[2] == "00"))
        mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[3] == "00"))
        mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[2] == "00"))
        mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[1] == "00"))
        mult.append((weigh_bit_slice[3] == "00") or (act_bit_slice[0] == "00"))
        mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[1] == "00"))
        mult.append((weigh_bit_slice[2] == "00") or (act_bit_slice[0] == "00"))
        mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[3] == "00"))
        mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[2] == "00"))
        mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[3] == "00"))
        mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[2] == "00"))
        mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[1] == "00"))
        mult.append((weigh_bit_slice[1] == "00") or (act_bit_slice[0] == "00"))
        mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[1] == "00"))
        mult.append((weigh_bit_slice[0] == "00") or (act_bit_slice[0] == "00"))
        count = 0
        for i in range(0, 16):
            if mult[i] == True:
                count = count + 1
            if ((i + 1) % 4) == 0:
                if count < 2:
                    flag.append(False)
                else:
                    flag.append(True)
                count = 0

    return flag

def statistic_one_multipy(weight_data, act_data, weight_bit_width, act_bit_width):
    weight_count_zero_slice = 0
    act_count_zero_slice = 0
    weight_data_bin = format(weight_data & 0xff, f'0{weight_bit_width}b')
    act_data_bin = format(act_data & 0xff, f'0{act_bit_width}b')
    # print(weight_data_bin)
    # print(act_data_bin)

    weight_data_l = split_string_by_length(weight_data_bin, 2)
    act_data_l = split_string_by_length(act_data_bin, 2)
    c = 0
    for i in range(len(weight_data_l) - 1, -1, -1):
        if i < len(weight_data_l) - 1:
            high = int(weight_data_l[i + 1][0])
            c = c | high
            next_value = int(weight_data_l[i], 2) + c
            # print(next_value)
            weight_data_l[i] = format(next_value, f'02b')[-2:]
            c = int(format(next_value, f'02b')[:-1], 2)
        if weight_data_l[i] == "00":
            weight_count_zero_slice = weight_count_zero_slice + 1

    c = 0
    for i in range(len(act_data_l) - 1, -1, -1):
        if i < len(act_data_l) - 1:
            high = int(act_data_l[i + 1][0])
            c = c | high
            next_value = int(act_data_l[i], 2) + c
            # print(next_value)
            act_data_l[i] = format(next_value, f'02b')[-2:]
            c = int(format(next_value, f'02b')[:-1], 2)
        if act_data_l[i] == "00":
            act_count_zero_slice = act_count_zero_slice + 1
    # print(len(weight_data_l))
    # print(len(act_data_l))
    count_zero_slice = {"weight" : weight_count_zero_slice, "activation" : act_count_zero_slice}
    zero_multiply = count_zero_multiply(weight_data_l, act_data_l, 8, 8)

    return str(zero_multiply)


import torch


def analyse_linear(input_tensor, weight, bias=None):
    with open('zero_slice.txt', 'a') as file:
        file.write("\n")
        file.write("layer: Linear\n")
    batch_size, in_features = input_tensor.shape
    out_features, _ = weight.shape
    # 初始化输出张量
    output = torch.zeros(batch_size, out_features)

    for b in range(batch_size):  # 遍历每个样本
        for o in range(out_features):  # 遍历输出维度
            for i in range(in_features):  # 遍历输入维度
                input_value = int(input_tensor[b, i])
                weight_value = int(weight[o, i])  # 逐元素乘法
                if not weight_value == 0:
                    with open('zero_slice.txt', 'a') as file:
                        # file.write(str(weight_value))
                        file.write(statistic_one_multipy(weight_value, input_value, 8, 8))
                        file.write(",")

def analyze_convolution(input_tensor, weight, stride=(1, 1), padding=(0, 0)):
    with open('zero_slice.txt', 'a') as file:
        file.write("\n")
        file.write("layer: Conv\n")

    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, K_h, K_w = weight.shape

    # 解析 padding 和 stride
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    # 计算输出尺寸
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

    # 修正 padding 4D 格式
    input_padded = torch.nn.functional.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    # 遍历每个样本
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # 计算输入窗口的起始位置
                    h_start, w_start = h * stride_h, w * stride_w

                    # 遍历每个输入通道和 kernel 位置
                    for c_in in range(C_in):
                        for i in range(K_h):
                            for j in range(K_w):
                                # 确保索引在合法范围内
                                h_idx, w_idx = h_start + i, w_start + j
                                if 0 <= h_idx < H_in + 2 * pad_h and 0 <= w_idx < W_in + 2 * pad_w:
                                    input_value = int(input_padded[n, c_in, h_idx, w_idx])
                                    weight_value = int(weight[c_out, c_in, i, j])
                                    if not weight_value == 0:
                                        with open('zero_slice.txt', 'a') as file:
                                            # file.write(str(weight_value))
                                            file.write(statistic_one_multipy(weight_value, input_value, 8, 8))
                                            file.write(",")

# input_tensor = torch.randint(-127, 128, (64, 64, 56, 56), dtype=torch.float32)
# weight = torch.randint(-127, 128, (64, 64, 1, 1), dtype=torch.float32)
#
#
# # 调用函数
# analyze_convolution(input_tensor, weight, stride=(1, 1), padding=(1, 1))
# import random
# print(statistic_one_multipy(random.randint(-127, 128), random.randint(-127, 128), 8, 8))









