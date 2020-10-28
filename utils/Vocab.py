# author: yx (zhuang xu's fork)
# date: 2020/9/2 14:03
import math
import numpy as np
from jieba.posseg import cut

stop_tags = ['m', 'uv', 'r', 'x', 'd', 'uj', 'u', 'c', 'p', 'o', 'ul', 'q', 'f']

class Vocab:
    def __init__(self, cfg):
        self.word2idf = {}      # doc contain the word
        self.word2f = {}        # word frequency
        self.word2index = {}    # index in document vector
        self.num_docs = 10.0
        self.stopwords = set()
        self.min_count = cfg.SOLVER.MIN_COUNT
        self.min_freq = cfg.SOLVER.MIN_FREQ
        self.total_count = .0
        if cfg.STOPWORDS_FILE is not None:
            with open(cfg.STOPWORDS_FILE, mode='r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    self.stopwords.add(line.strip('\n'))

    def __len__(self):
        return len(self.word2index)

    def update(self, content):
        self.num_docs += 1
        w_list = self._cut(content)

        # update term count in the whole corpus
        for w in w_list:
            self.total_count += 1
            if self.word2f.get(w) is not None:
                self.word2f[w] += 1.0
                pass
            else:
                self.word2f[w] = 1.0
                pass

        # update inverse term count (number of doc containing the term) in the whole corpus
        w_set = set(w_list)
        for w in w_set:
            if self.word2idf.get(w) is not None:
                self.word2idf[w] += 1.0
                pass
            else:
                self.word2idf[w] = 1.0
                pass
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

    def to_tfiwf(self, content):
        w_list = self._cut(content)
        vector = np.zeros(len(self.word2index))
        c = 0
        for w in w_list:
            if self.word2index.get(w) is not None:
                vector[self.word2index[w]] += 1
                c += 1
                pass
            pass
        if c == 0:
            return None
        else:
            # TF part implementation
            vector = vector / c

            for w in w_list:
                if self.word2index.get(w) is not None:
                    # TF-IWF
                    vector[self.word2index[w]] *= math.log(self.total_count / self.word2f[w], 10)
                    pass
                pass
            return vector

    def to_tfidf(self, content):
        w_list = self._cut(content)
        vector = np.zeros(len(self.word2index))
        c = 0

        for w in w_list:
            if self.word2index.get(w) is not None:
                vector[self.word2index[w]] += 1
                c += 1
                pass
            pass
        if c == 0:
            return None
        else:
            vector = vector / c
            for w in w_list:
                if self.word2index.get(w) is not None:
                    vector[self.word2index[w]] = math.log(self.total_count / self.word2idf[w], 10)
                    pass
                pass
            return vector

    def to_incremental_tfidf(self, content):
        """
        Incremental TF-IDF model is widely applied to term weight calculation in TDT
        :param content: news content
        :return: an n-dimension vector, which represent each news d coming at time t
        """
        w_list = self._cut(content)
        w_set = set(w_list)

        # W_t represents the total number of term appearance before time t
        W_t = sum(self.word2f.values()) - len(w_list)

        denominator = 0.0

        # each content d coming at time t is represented as an n-dim vector,
        # where n is the number of distinct terms in all content which have been processed .
        vector = np.zeros(len(self.word2index))


        # the denominator is a cumulative process
        for w in w_set:
            if self.word2index.get(w) is not None:
                # tf_d_w means how many times w appears in content d
                tf_d_w = w_list.count(w)

                # wf_t_w represents the number of times term w appears
                wf_t_w = self.word2f[w]

                vector[self.word2index[w]] = (tf_d_w * math.log((W_t + 1) / (wf_t_w + 0.5), 10))

                denominator += vector[self.word2index[w]] ** 2

        denominator = math.sqrt(denominator)

        for i in range(len(vector)):
            vector[i] = vector[i] / denominator
            pass

        return vector


    @staticmethod
    def cosine_sim(x1, x2):
        if len(x1) < len(x2):
            np.append(np.zeros(len(x2) - len(x1)))
            pass
        elif len(x2) < len(x1):
            np.append(np.zeros(len(x1) - len(x2)))
            pass

        sim = np.dot(x1, x2)
        sim = sim / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return sim

    def cosine_sim_decay(x1, l1, x2, l2):
        if len(x1) < len(x2):
            np.append(np.zeros(len(x2) - len(x1)))
            pass
        elif len(x2) < len(x1):
            np.append(np.zeros(len(x1) - len(x2)))
            pass

        sim = np.dot(x1, x2)
        decay = (0.95 ** l1) * (0.95 ** l2)
        sim_decay = sim / (np.linalg.norm(x1) * np.linalg.norm(x2)) * decay
        return sim_decay, sim

    def _cut(self, text):
        words = cut(text)

        w_list = []
        for word, tag in words:
            if tag not in stop_tags and len(word) > 1 and word not in self.stopwords:
                w_list.append(word)
                pass
            pass
        return w_list

