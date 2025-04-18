import math
import operator
import time

import torch as t


__all__ = ['validate', 'PerformanceScoreboard']



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def validate(data_loader, model, args):
#     # losses = AverageMeter()
#     # top1 = AverageMeter()
#     # top5 = AverageMeter()
#     # batch_time = AverageMeter()
#
#     total_sample = len(data_loader.sampler)
#     batch_size = data_loader.batch_size
#     steps_per_epoch = math.ceil(total_sample / batch_size)
#
#     model.eval()
#     # end_time = time.time()
#     idx = 0
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         with t.no_grad():
#             idx = idx + 1
#             print(idx)
#             inputs = inputs.to(args.device.type)
#             targets = targets.to(args.device.type)
#
#             outputs = model(inputs)
#             with open('conv_8x8.txt', 'a') as f:
#                 f.write('\n')
#             with open('linear_8x8.txt', 'a') as f:
#                 f.write('\n')
#             # loss = criterion(outputs, targets)
#
#             acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#             print(acc1)
#             print(acc5)
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))
            # batch_time.update(time.time() - end_time)
            # end_time = time.time()

            # if (batch_idx + 1) % args.log.print_freq == 0:
                # for m in monitors:
                #     m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                #         'Loss': losses,
                #         'Top1': top1,
                #         'Top5': top5,
                #         'BatchTime': batch_time
                #     })


    # logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    # return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, args):
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0

    # for images, labels in data_loader:
    #     images, labels = images.to(args.device.type), labels.to(args.device.type)
    #     outputs = model(images)
    #     print("Output logits:", outputs[0][:5])
    #     print("Target label:", labels[0])
    #     break

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)
            t.cuda.empty_cache()
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # 累加准确率
            batch_size = inputs.size(0)
            top1_correct += acc1.item() * batch_size / 100.0
            top5_correct += acc5.item() * batch_size / 100.0
            total += batch_size

            print(f"Step {batch_idx+1}/{len(data_loader)} - Top1: {acc1.item():.2f}% - Top5: {acc5.item():.2f}%")

            # 每步清空记录文件
            # with open('conv_8x8.txt', 'a') as f:
            #     f.write('\n')
            # with open('linear_8x8.txt', 'a') as f:
            #     f.write('\n')

    final_top1 = 100.0 * top1_correct / total
    final_top5 = 100.0 * top5_correct / total

    print(f"\nFinal Top-1 Accuracy: {final_top1:.2f}%")
    print(f"Final Top-5 Accuracy: {final_top5:.2f}%")

    return final_top1, final_top5


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
