# author: yx (zhuang xu's fork)
# date: 2020/9/2 14:32

class Point:
    def __init__(self, text, vector):
        self.text = text
        self.vec = vector

    def __len__(self):
        if self.vec is None:
            return 0
        else:
            return len(self.vec)

    def update(self, voc):
        self.vec = voc.to_incremental_tfidf(self.text)
        pass
