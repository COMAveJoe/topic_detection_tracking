# author: yx
# date: 2020/9/3 14:40
from torch.utils import data
from .news_flow import NewsFlow

def build_dataset(cfg):
    datasets = NewsFlow(cfg)
    return datasets
