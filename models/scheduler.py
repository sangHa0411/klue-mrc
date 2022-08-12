
import torch

class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, warmup_ratio, last_epoch=-1):

        def lr_lambda(step):
            warmup_steps = int(total_steps * warmup_ratio)            
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            else :
                return (float(total_steps) - float(step)) / float(total_steps - warmup_steps)

        super(LinearWarmupScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)