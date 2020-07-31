import torch


class Metrics:
    ''' Metrics computes accuracy/precision/recall/confusion_matrix with batch updates. '''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []

    def update(self, y, t):
        ''' Accuracy with batch updates.

        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].

        Returns:
          (tensor): class accuracy.
        '''
        self.y.append(y)
        self.t.append(t)

    def _process(self, y, t):
        ''' Compute TP, FP, FN, TN.

        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].

        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        tp = torch.empty(self.num_classes)
        fp = torch.empty(self.num_classes)
        fn = torch.empty(self.num_classes)
        tn = torch.empty(self.num_classes)
        for i in range(self.num_classes):
            tp[i] = ((y == i) & (t == i)).sum().item()
            fp[i] = ((y == i) & (t != i)).sum().item()
            fn[i] = ((y != i) & (t == i)).sum().item()
            tn[i] = ((y != i) & (t != i)).sum().item()
        return tp, fp, fn, tn

    def accuracy(self, reduction='mean'):
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        if reduction == 'none':
            acc = tp / (tp + fn)
        else:
            acc = tp.sum() / (tp + fn).sum()
        return acc


def test():
    import pytorch_lightning.metrics.functional as M
    m = Metrics(3)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    t = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    m.update(y, t)

    print(m.accuracy('none'))
    print(M.accuracy(y, t, 3, 'none'))

    print()
    print(m.accuracy('mean'))
    print(M.accuracy(y, t, 3, 'elementwise_mean'))


if __name__ == '__main__':
    test()
