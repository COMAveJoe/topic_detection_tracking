# author: yx
# date: 2020/9/2 15:00
from tqdm import trange
from utils.TFIDFClustering import TFIDFClustring
from utils.Vocab import Vocab
from tools.get_topic_keywords import Keyword
from tools.get_text_summary import SummaryTxt
from data.build import build_dataset

def train(cfg):
    """
    training begin
    :param cfg: config file
    :return:
    """
    datasets = build_dataset(cfg)
    algo = TFIDFClustring(cfg)
    vocab = Vocab(cfg)
    summary = SummaryTxt(cfg)
    keyword = Keyword(cfg, summary)

    processed_news_num = 0
    batch_size = cfg.SOLVER.BATCH_SIZE

    print('start training:')
    for seg_id in trange(0, datasets.file_num, batch_size):
        seg = []
        for batch_idx in range(batch_size):
            batch, seg_size = datasets.getitem(seg_id + batch_idx)
            seg.extend(batch)
            processed_news_num += seg_size

        algo.run(segments=seg, vocab=vocab, seg_id=seg_id, keyword=keyword, summary=summary)
        # keyword.update_per_seg(new_updated_topic=new_updated_topic)
        print("seg idx: {}. processed news: {}".format(seg_id, processed_news_num))
        pass