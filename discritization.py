import torch

class SID:
    def __init__(self, dataset):
        super(SID, self).__init__()
        if dataset == 'kitti':
            alpha = 0.001
            beta = 80.0
        elif dataset == 'nyu':
            alpha = 0.7113
            beta = 9.9955
        
        K = 80.0
            
        self.alpha = torch.tensor(alpha).cuda()
        self.beta = torch.tensor(beta).cuda()
        self.K = torch.tensor(K).cuda()
        
    def labels2depth(self, labels):
        depth = self.alpha * (self.beta / self.alpha) ** (labels.float() / self.K)
        return depth.float()

    
    def depth2labels(self, depth):
        labels = self.K * torch.log(depth / self.alpha) / torch.log(self.beta / self.alpha)
        return labels.cuda().round().int()
