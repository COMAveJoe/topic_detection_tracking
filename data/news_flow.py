# author: yx
# date: 2020/9/2
import os


def get_daily_news_file_names(cfg):
    """
    get daily news record file names
    :param cfg: config
    :return: file names and file num
    """
    flow_data_dir = cfg.DATASET.FLOW_DATA_CATALOG
    file_names = [os.path.join(flow_data_dir, file_name) for file_name in os.listdir(flow_data_dir)]
    file_names.sort()
    file_num = len(file_names)
    return file_names, file_num

def get_daily_news_content(file_name):
    """
    get daily news content
    :param file_name: the file name which want to get the content
    :return: daily news content, news num
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines, len(lines)


class NewsFlow():
    """
    we set the day as the basic flow unit
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.file_names, self.file_num = get_daily_news_file_names(cfg)
        pass

    def len(self):
        return self.file_num

    def getitem(self, item):
        daily_news_content = get_daily_news_content(self.file_names[item])
        return daily_news_content
