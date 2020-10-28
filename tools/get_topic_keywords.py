# author: yx
# date: 2020/9/2 16:20
import os
import heapq
import numpy as np
import jieba.analyse
import jieba.posseg
from tqdm import trange
from .get_topic_event import DbscanClustering
from utils.Vocab import Vocab

import jpype
from pyhanlp import *

np.seterr(divide='ignore', invalid='ignore')

stop_tags = ['m', 'uv', 'r', 'x', 'd', 'uj', 'u', 'c', 'p', 'o', 'ul', 'q', 'f']

class Keyword(Vocab):
    """
    The Keywords class get a Top-k keywords per topic. This is useful for online calculating of tf-idf
    """
    def __init__(self, cfg, summary):
        Vocab.__init__(self, cfg)
        self.file_path = cfg.OUTPUT_DIR
        self.summary = summary
        pass

    def filter(self):
        self.word2index = {}
        count = 0
        for key in self.word2f:
            if self.word2f[key] >= self.min_count and self.word2f[key] / self.total_count >= self.min_freq:
                self.word2index[key] = count
                count += 1
                pass
            pass

        pass

    def update(self, topic_name):
        self.word2f = {}
        self.total_count = 0
        with open(os.path.join(self.file_path, topic_name), 'r+', encoding='utf-8') as f:
            try:
                cont_list = f.readlines()
                content = ''.join(cont_list[1:])
            except:
                # todo: find the reason why we can't decode some content
                # catch the error 'utf-8' codec can't decode byte
                pass
            pass
        w_list = self._cut(content)
        for w in w_list:
            self.total_count += 1
            if self.word2f.get(w) is not None:
                self.word2f[w] += 1.0
                pass
            else:
                self.word2f[w] = 1.0
                pass
            pass
        self.filter()
        try:
            # TEXT-RANK
            # keywords = jieba.analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn'))

            # TF-IDF
            keywords_tfidf = jieba.analyse.extract_tags(content, topK=5, withWeight=False, allowPOS=('ns', 'n', 'vn'))

            # TF-IWF
            vec = self.to_tfiwf(content)

            # get topic content summary
            summary_text = self.summary.summaryTopNtxt(content)
            # summary_text_2 = HanLP.extractSummary(content, 5)

            top_k_index = list(map(vec.tolist().index, heapq.nlargest(min(len(vec), 10), vec)))
            top_k_index = list(set(top_k_index))
            tmp_dic = {v: k for k, v in self.word2index.items()}
            keywords_tfiwf = [tmp_dic[idx] for idx in top_k_index]

            # fuse tfidf keywords and tfiwf keywords
            keywords_tfidf.extend(keywords_tfiwf)
            keywords = list(set(keywords_tfidf))

            with open(os.path.join(self.file_path, topic_name), 'r+', encoding='utf-8') as f:
                f.seek(0, 0)
                f.write('keywords: [')
                f.write(' '.join(keywords) + '] ')
                f.write('summary: {')
                f.write(summary_text + '}')
                f.write('\n' + content)
        except ValueError as e:
            print(e)
            pass
        pass


    def update_per_seg(self, new_updated_topic):
        """
        get Top-k keywords of topic by using manual TF-IWF,
        get event cluster by using DBScan,
        get event tag by text rank
        when one seg finish
        :param new_updated_topic:
        :return:
        """
        print('analyse top-k keywords, event cluster and event tag: ')
        for i in range(len(new_updated_topic)):
            topic_name = new_updated_topic[i]

            # get the num of news in a topic
            num_points = int(topic_name.split('_')[1])

            self.word2f = {}
            self.total_count = 0

            with open(os.path.join(self.file_path, topic_name), 'r+', encoding='utf-8') as f:
                try:
                    cont_list = f.readlines()
                    content = ''.join(cont_list[1:])
                except:
                    # todo: find the reason why we can't decode some content
                    # catch the error 'utf-8' codec can't decode byte
                    continue
                pass

            w_list = self._cut(content)
            # dbscan = DbscanClustering(self.stopwords)
            for w in w_list:
                self.total_count += 1
                if self.word2f.get(w) is not None:
                    self.word2f[w] += 1.0
                    pass
                else:
                    self.word2f[w] = 1.0
                    pass
                pass
            self.filter()
            try:
                vec = self.to_tfiwf(content)

                top_k_index = list(map(vec.tolist().index, heapq.nlargest(min(len(vec), 10), vec)))
                top_k_index = list(set(top_k_index))
                tmp_dic = {v: k for k, v in self.word2index.items()}
                keywords = [tmp_dic[idx] for idx in top_k_index]

                with open(os.path.join(self.file_path, topic_name), 'r+', encoding='utf-8') as f:
                    f.seek(0, 0)
                    f.write(' '.join(keywords) + ' ')

                    # if num_points > 1:
                    #     result = dbscan.dbscan(os.path.join(self.file_path, topic_name), eps=0.05, min_samples=2)
                    #     for key, value in result.items():
                    #         f.write('{} : {},'.format(key + 1, str(value)))
                    #         pass
                    #     pass
                    # else:
                    #     event_tag = jieba.analyse.textrank(content, topK=5, withWeight=False,
                    #                                        allowPOS=('ns', 'n', 'vn'))
                    #     f.write('{} : {} {},'.format(0, '1', event_tag))
                    f.write('\n' + content)
            except:
                pass
            pass
        pass