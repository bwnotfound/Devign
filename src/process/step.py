import torch
from ..utils.objects import stats


def softmax_accuracy(probs, all_labels):
    preds = torch.argmax(probs, dim=-1)
    acc = (preds == all_labels).sum()
    acc = acc / max(all_labels.shape[0], 1)
    positive_num = (preds == 1).sum()
    return acc, positive_num.item(), all_labels.shape[0]


class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer

    def __call__(self, i, x, y):
        out = self.model(x)
        loss = self.criterion(out, y)
        acc, positive_num, total_num = softmax_accuracy(out, y)

        if self.model.training:
            # calculates the gradient
            loss.backward()
            # and performs a parameter update based on it
            self.optimizer.step()
            # clears old gradients from the last step
            self.optimizer.zero_grad()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(
            out.tolist(), loss.item(), acc.item(), y.tolist(), positive_num, total_num
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
