# author: yx (zhuang xu's fork)
# date: 2020/9/2 14:40

from .Vocab import Vocab

class Cluster:
    def __init__(self, point, cluster_id, seg_id):
        self.id = cluster_id
        self.points = [point]
        self.center = point.vec
        self.newpoints = [point]     # keep a copy of new added points in last running iteration
        self.to_eliminate = False
        self.seg_id = seg_id        # denotes the time (number of segments) of creation of the cluster
        self.last_seg_id = seg_id   # denotes the time (number of segments) of creation of the cluster

        '''
        keep the number of added points in past. This cannot be inferred from len(self.points) because som added
        points in the list may be removed for some reasons (e.g. all of its words are not contained bu current
        vocabulary)
        '''
        self.num_points = 1

    def add_point(self, point, seg_id):
        self.last_seg_id = seg_id
        self.points.append(point)
        self.num_points += 1
        s = self.points[0].vec
        for j in range(1, len(self.points)):
            s += self.points[j].vec
            pass
        self.center = s / len(self.points)
        self.newpoints.append(point)
        pass

    def expand_cluster(self, cluster, seg_id):
        """
        the candidates topics should be update old topics
        :param cluster: the candidates topics
        :param seg_id: seg id
        :return:
        """
        self.last_seg_id = seg_id
        self.points.extend(cluster.points)
        self.num_points += cluster.num_points
        s = self.points[0].vec
        for j in range(1, len(self.points)):
            s += self.points[j].vec
            pass
        self.center = s / len(self.points)
        self.newpoints.extend(cluster.points)
        pass

    def cosine_sim(self, point):
        return Vocab.cosine_sim(self.center, point.vec)

    def cosine_sim_decay(self, cluster):
        """
        cal the cosine similarity between two topics, cosine_sim_decay = cosine_sim * decay_1 * decay_2
        we add the cosine decay as 0.95 ** n, in order to control the topic size
        :param cluster: each topic in time t
        :return: cosine similarity
        """
        return Vocab.cosine_sim_decay(self.center, self.num_points, cluster.center, cluster.num_points)

    def update(self, voc):
        for point in self.points:
            point.update(voc)
            pass
        self.points = list(filter(None, self.points))

        if len(self.points) == 0:
            self.to_eliminate = True
            return

        s = self.points[0].vec
        for j in range(1, len(self.points)):
            s += self.points[j].vec
            pass
        if type(s) is None:
            print(1)
            pass
        self.center = s / len(self.points)


