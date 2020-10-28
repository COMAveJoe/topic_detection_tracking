# author: yx
# date: 2020/9/8 15:45
import math
import numpy as np
from itertools import combinations
from jieba.posseg import cut

stop_tags = ['m', 'uv', 'r', 'x', 'd', 'uj', 'u', 'c', 'p', 'o', 'ul', 'q', 'f']

class wPair:
    def __init__(self, cfg):
        self.word2idf = {}      # doc contain the word
        self.word2f = {}        # word frequency
        self.pair2index = {}    # index in document vector
        self.num_docs = 10.0
        self.word_pairs = {}
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

    def _split_sentences(self, texts):
        '''
        split texts into sentences
        :param texts: text
        :return:
        '''
        split_str = '.!?。！？,，'
        start = 0
        index = 0
        sentences = []
        for text in texts:
            if text in split_str:
                sentences.append(texts[start:index + 1])
                start = index + 1
            index += 1
        if start < len(texts):
            sentences.append(texts[start:])

        return sentences

    def update(self, content):
        self.num_docs += 1

        sentences = self._split_sentences(content)

        for sentence in sentences:
            w_list = self._cut(sentence)

            # build word pairs dict
            for item in combinations(w_list, 2):
                if self.word_pairs.get(item) is None:
                    self.word_pairs[item] = 0.0
                    pass
                pass

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
        self.pair2index = {}
        count = 0
        for pair in self.word_pairs.keys():
            self.pair2index[pair] = count
            count += 1
            pass
        pass

    def to_wp_tfidf(self, content):
        # split content into sentences
        sentences = self._split_sentences(content)
        # vector length is the pair nums
        vector = np.zeros(len(self.pair2index))
        c = 0
        pairs = []
        for sentence in sentences:
            # cut each sentence
            w_list = self._cut(sentence)

            for pair in combinations(w_list, 2):
                if self.pair2index.get(pair) is not None:
                    vector[self.pair2index[pair]] += 1
                    c += 1
                if pair not in pairs:
                    pairs.append(pair)
        if c == 0:
            return None
        else:
            vector = vector / c
            for pair in pairs:
                if self.pair2index.get(pair) is not None:
                    # w(t1, t2) = V_tf_idf(t1) + V_tf_idf(t2)
                    vector[self.pair2index[pair]] *= \
                        math.log(self.total_count / self.word2idf[pair[0]], 10) + math.log(self.total_count / self.word2idf[pair[1]], 10)
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

