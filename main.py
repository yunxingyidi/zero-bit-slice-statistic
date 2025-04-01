from pathlib import Path

import torch as t
import yaml

import process
import quantization
import util
import torchvision.models as models


def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # 设置设备
    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        t.cuda.set_device(args.device.gpu[0])
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # 初始化数据加载器（仅加载测试集）
    test_loader = util.load_data(args.dataloader)

    # 加载模型
    model = models.resnet50(pretrained=True)

    modules_to_replace = quantization.find_modules_to_quantize(model, args.quan)
    model = quantization.replace_module_by_names(model, modules_to_replace)
    # t.save(model.state_dict(), 'model.pth')
    # checkpoint = t.load('model.pth')
    # # print(checkpoint.keys())
    # print(checkpoint['conv1.weight'])  # 打印 state_dict 的键

    if args.device.gpu:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)

    process.validate(test_loader, model, args)



if __name__ == "__main__":
    main()