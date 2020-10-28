# author: yx
# date: 2020/9/7 15:15
import nltk
import numpy
import jieba
import codecs
import os

class SummaryTxt:
    def __init__(self, cfg):
        stop_words_path = cfg.STOPWORDS_FILE
        # key
        self.N = 100
        # words distance
        self.CLUSTER_THRESHOLD = 5
        # top n sentence
        self.TOP_SENTENCES = 5
        self.stop_words = {}
        # load stop words
        if os.path.exists(stop_words_path):
            stop_list = [line.strip() for line in codecs.open(stop_words_path, 'r', encoding='utf8').readlines()]
            self.stop_words = {}.fromkeys(stop_list)


    def _split_sentences(self, texts):
        '''
        split texts into sentences
        :param texts: text
        :return:
        '''
        split_str = '.!?。！？,， '
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

    def _score_sentences(self,sentences, top_n_words):
        """
        score sentences by top n keywords
        :param sentences: sentences list
        :param top_n_words: top n keywords list
        :return:
        """
        scores = []
        sentence_idx = -1
        for s in [list(jieba.cut(s)) for s in sentences]:
            sentence_idx += 1
            word_idx = []
            for w in top_n_words:
                try:
                    word_idx.append(s.index(w))     # the index that keyword appears in the sentence
                except ValueError:                  # word is not in the sentence
                    pass
            word_idx.sort()
            if len(word_idx) == 0:
                continue
            # for two nearby words, use the word's position index and distance threshold cal clusters
            clusters = []
            cluster = [word_idx[0]]
            i = 1
            while i < len(word_idx):
                if word_idx[i] - word_idx[i - 1] < self.CLUSTER_THRESHOLD:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)
            # score clusters
            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                score = 1.0 * significant_words_in_cluster * significant_words_in_cluster / total_words_in_cluster
                if score > max_cluster_score:
                    max_cluster_score = score
            scores.append((sentence_idx, max_cluster_score))
        return scores

    def summaryScoredtxt(self,text):
        # split content into sentences
        sentences = self._split_sentences(text)

        # tokenize words
        words = [w for sentence in sentences for w in jieba.cut(sentence) if w not in self.stop_words if
                 len(w) > 1 and w != '\t']

        # word frequency
        word_fre = nltk.FreqDist(words)

        # the top n words
        top_n_words = [w[0] for w in sorted(word_fre.items(), key=lambda d: d[1], reverse=True)][:self.N]

        # score sentence
        scored_sentences = self._score_sentences(sentences, top_n_words)

        # use average and standard deviation to filter non-important sentences
        avg = numpy.mean([s[1] for s in scored_sentences])  # average
        std = numpy.std([s[1] for s in scored_sentences])  # standard
        summarySentences = []

        n = 0
        for (sent_idx, score) in scored_sentences:
            if score > (avg + 0.5 * std):
                tmp_sentence = sentences[sent_idx].replace('\n', '')
                if tmp_sentence not in summarySentences:
                    summarySentences.append(sentences[sent_idx].replace('\n', ''))
                    n += 1
                    if n > self.TOP_SENTENCES:
                        break
                        pass
                # print(sentences[sent_idx])
        return ''.join(summarySentences)

    def summaryTopNtxt(self, text):
        # split content into sentences
        sentences = self._split_sentences(text)

        # words tokenize
        words = [w for sentence in sentences for w in jieba.cut(sentence) if w not in self.stop_words if
                 len(w) > 1 and w != '\t']

        # word frequency
        word_fre = nltk.FreqDist(words)

        # top n words
        top_n_words = [w[0] for w in sorted(word_fre.items(), key=lambda d: d[1], reverse=True)][:self.N]

        # score sentence
        scored_sentences = self._score_sentences(sentences, top_n_words)

        top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-self.N:]
        top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
        summarySentences = []

        n = 0
        for (idx, score) in top_n_scored:
            # print(sentences[idx])
            tmp_sentence = sentences[idx].replace('\n', '')
            if tmp_sentence not in summarySentences:
                summarySentences.append(sentences[idx].replace('\n', ''))
                n += 1
                if n > self.TOP_SENTENCES:
                    break
                    pass
                pass
            pass

        return ''.join(summarySentences)