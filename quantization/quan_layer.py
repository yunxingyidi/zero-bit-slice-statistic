import torch as t
from .statistic import *
conv_zero_slice = 0
linear_zero_slice = 0

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None,  quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight, weight_scale = self.quan_w_fn(self.weight)
        quantized_act, act_scale = self.quan_a_fn(x)

        # with open("tensor_full.txt", "a") as f:
        #     f.write(quantized_act.__repr__())  # 使用 __repr__ 避免中间省略

        # num_greater = ((quantized_weight > 64)).sum().item()
        # total = quantized_weight.numel()
        # ratio = num_greater / total
        # print(f"大于20的比例是: {ratio:.2%}")

        # output = self._conv_forward(quantized_act, quantized_weight, bias=None)
        output = compute_convolution(quantized_act, quantized_weight, stride=self.stride, padding=self.padding)
        scale = weight_scale.view(1, -1, 1, 1) * act_scale.view(-1, 1, 1, 1)  # shape: [B, C, 1, 1]
        output_fp = output * scale
        if self.bias is not None:
            output_fp += self.bias.view(1, -1, 1, 1)
        # print("conv_\n")
        return output_fp
        # analyze_convolution(quantized_act, quantized_weight, self.stride, self.padding)
        # return self._conv_forward(quantized_act, quantized_weight, bias=None)


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight, weight_scale = self.quan_w_fn(self.weight)
        quantized_act, act_scale = self.quan_a_fn(x)
        # analyse_linear(quantized_act, quantized_weight)
        # output = t.nn.functional.linear(quantized_act, quantized_weight, self.bias)  # [B, out_features]
        output = compute_linear(quantized_act, quantized_weight.T, self.bias)
        # print(output)
        scale = act_scale.view(-1, 1) * weight_scale.view(1, -1)
        output_fp = output * scale
        return output_fp


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear
}
