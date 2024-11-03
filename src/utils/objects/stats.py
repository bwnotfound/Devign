import dataclasses
from dataclasses import dataclass
from typing import List


class Stat:
    def __init__(self, outs=None, loss=0.0, acc=0.0, labels=None, positive_num=0, total_num=0):
        if labels is None:
            labels = []
        if outs is None:
            outs = []
        self.outs = outs
        self.labels = labels
        self.loss = loss
        self.acc = acc
        self.positive_num = positive_num
        self.total_num = total_num

    @property
    def positive_ratio(self):
        return self.positive_num / max(self.total_num, 1)

    def __add__(self, other):
        return Stat(
            self.outs + other.outs,
            self.loss + other.loss,
            self.acc + other.acc,
            self.labels + other.labels,
            self.positive_num + other.positive_num,
            self.total_num + other.total_num,
        )

    def __str__(self):
        return f"Loss: {round(self.loss, 4)}; Acc: {round(self.acc, 4)}; Positive Ratio: {round(self.positive_ratio, 4)}"


@dataclass
class Stats:
    name: str
    results: List[Stat] = dataclasses.field(default_factory=list)
    total: Stat = Stat()

    def __call__(self, stat):
        self.total += stat
        self.results.append(stat)

    def __str__(self):
        return f"{self.name} {self.mean()}"

    def __len__(self):
        return len(self.results)

    def mean(self):
        res = Stat()
        res += self.total
        res.loss /= len(self)
        res.acc /= len(self)

        return res

    def loss(self):
        return self.mean().loss

    def acc(self):
        return self.mean().acc

    def outs(self):
        return self.total.outs

    def labels(self):
        return self.total.labels
