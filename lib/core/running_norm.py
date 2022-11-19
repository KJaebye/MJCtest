import torch


class RunningNorm(torch.nn.Module):
    """
    y = (x-mean)/std
    using running estimates of mean, std
    """

    def __init__(self, dim, de_mean=True, de_std=True, clip=5.0):
        super().__init__()
        self.dim = dim
        self.de_mean = de_mean
        self.de_std = de_std
        self.clip = clip
        self.register_buffer('n', torch.tensor(0, dtype=torch.long))
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def update(self, x):
        var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False)
        m = x.shape[0]
        w = self.n.to(x.dtype) / (m + self.n).to(x.dtype)
        self.var[:] = w * self.var + (1 - w) * var_x + w * (1 - w) * (mean_x - self.mean).pow(2)
        self.mean[:] = w * self.mean + (1 - w) * mean_x
        self.std[:] = torch.sqrt(self.var)
        self.n += m

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.update(x)
        if self.n > 0:
            if self.de_mean:
                x = x - self.mean
            if self.de_std:
                x = x / (self.std + 1e-8)
            if self.clip:
                x = torch.clamp(x, -self.clip, self.clip)
        return x
