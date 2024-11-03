from ..utils.objects import stats

from tqdm import tqdm

class LoaderStep:
    def __init__(self, name, data_loader, device):
        self.name = name
        self.loader = data_loader
        self.size = len(data_loader)
        self.device = device

    def __call__(self, step):
        self.stats = stats.Stats(self.name)

        for i, batch in tqdm(enumerate(self.loader), total=len(self.loader), leave=False):
            batch.to(self.device)
            stat: stats.Stat = step(i, batch, batch.y)
            self.stats(stat)

        return self.stats
