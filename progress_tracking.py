import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)
        return avg
    
    def log(self, logger, epoch, stage="Train"):
        avg = self.average()
        logger.add_scalar(stage + '/RMSE', avg.rmse, epoch)
        logger.add_scalar(stage + '/rml', avg.absrel, epoch)
        logger.add_scalar(stage + '/Log10', avg.lg10, epoch)
        logger.add_scalar(stage + '/Delta1', avg.delta1, epoch)
        logger.add_scalar(stage + '/Delta2', avg.delta2, epoch)
        logger.add_scalar(stage + '/Delta3', avg.delta3, epoch)
        
        
class ImageBuilder(object):
    """
    Builds an image iteratively row by row where the columns are (input image, target depth map, output depth map).
    """
    def __init__(self):
        self.count = 0
        self.img_merge = None
        
    def has_image(self):
        return self.img_merge is not None
        
    def get_image(self):
        return torch.from_numpy(np.transpose(self.img_merge, (2, 0, 1)) / 255.0)
        
    def add_row(self, input, target, depth):
        if self.count == 0:
            self.img_merge = self.merge_into_row(input, target, depth)
        else:
            row = self.merge_into_row(input, target, depth)
            self.img_merge = np.vstack([self.img_merge, row])
            
        self.count += 1
    
    @staticmethod
    def colored_depthmap(depth, d_min=None, d_max=None):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C

    @staticmethod
    def merge_into_row(input, depth_target, depth_pred):
        rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
        depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
        depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

        d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
        d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
        depth_target_col = ImageBuilder.colored_depthmap(depth_target_cpu, d_min, d_max)
        depth_pred_col = ImageBuilder.colored_depthmap(depth_pred_cpu, d_min, d_max)
        img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

        return img_merge