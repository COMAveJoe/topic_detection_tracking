# author: yx (zhuang xu's fork)
# date: 2020/9/2 15:10
import os
import shutil
from tqdm import trange
from .Cluster import Cluster
from .Point import Point

class TFIDFClustring:
    def __init__(self, cfg):
        self.sim_p = cfg.SOLVER.SIM_P
        self.sim_t = cfg.SOLVER.SIM_T
        self.sim_i = cfg.SOLVER.SIM_I
        self.output_dir = cfg.OUTPUT_DIR
        self.history_dir = cfg.HISTORY_DIR
        self.seg_t = cfg.SOLVER.LIFE_CYCLE
        self.topic_candidates = []
        self.previous_topics = []
        self.new_updated_topic = []
        self.num_clusters = 0

    def update(self, vocab):
        for cluster in self.previous_topics:
            cluster.update(vocab)
            pass
        pass

    # step 1: we cluster news content into topics candidates
    def add_point(self, point, seg_id):
        if len(self.topic_candidates) == 0:
            # the first cluster
            self.topic_candidates.append(Cluster(point, cluster_id=self.num_clusters, seg_id=seg_id))
            self.num_clusters += 1
            pass
        else:
            max_sim = 0
            max_sim_cluster = None
            for c in self.topic_candidates:
                if c.to_eliminate:
                    continue

                sim = c.cosine_sim(point)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_cluster = c
            if max_sim > self.sim_p:
                max_sim_cluster.add_point(point, seg_id=seg_id)
                pass
            else:
                self.topic_candidates.append(Cluster(point, cluster_id=self.num_clusters, seg_id=seg_id))
                self.num_clusters += 1
                pass

    # step 2: compares those with previous topics and decide whether the candidates
    # should be used to updated or regards as new ones
    def update_old_topics(self, seg_id):
        """
        update previous topics
        :param seg_id: seg id
        :return: new updated topics
        """
        if len(self.previous_topics) == 0:
            # no previous topics, so set the current topics as previous topics directly
            self.previous_topics = self.topic_candidates[:]
            pass
        else:
            max_sim = 0
            max_sim_cluster = None
            # cal the cosine sim between topics candidates and previous topics orderly
            for c in self.topic_candidates:
                if c.to_eliminate:
                    continue
                    pass
                for h_c in self.previous_topics:
                    sim_decay, sim = c.cosine_sim_decay(h_c)
                    if sim_decay > max_sim:
                        max_sim = sim
                        max_sim_cluster = c
                        pass
                    pass
                if max_sim > self.sim_t:
                    max_sim_cluster.expand_cluster(c, seg_id)
                    pass
                else:
                    # put the new topic in previous topics list
                    self.previous_topics.append(c)
                    self.num_clusters += 1
                    pass
                pass
            pass
        pass

    def run(self, segments, vocab, seg_id, keyword, summary):
        self.new_updated_topic = []

        # empty list clusters, in order to restore new topic candidates
        self.topic_candidates = []

        for i in trange(len(segments)):
            # cal tf, idf and expand vocabulary table
            vocab.update(segments[i])
            pass
        # remove low frequency words
        vocab.filter()

        # update previous clusters, change the points' vector in previous topics have the same length as the current points
        self.update(vocab)

        for i in trange(len(segments)):
            # vec = vocab.to_tfidf(segments[i])
            vec = vocab.to_incremental_tfidf(segments[i])
            if vec is None:
                continue
            self.add_point(Point(segments[i], vec), seg_id=seg_id)
            pass

        # update previous topics, we store updated topics in self.history_clusters
        self.update_old_topics(seg_id)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            pass

        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            pass

        new_cluster_list = []
        for cluster in self.previous_topics:
            old_file = os.path.join(self.output_dir,
                                    'cluster_{}_{}.txt'.format(cluster.num_points - len(cluster.newpoints),
                                                               cluster.id))
            new_file = os.path.join(self.output_dir,
                                    'cluster_{}_{}.txt'.format(cluster.num_points, cluster.id))

            if len(cluster.newpoints) > 0:
                with open(old_file, mode='a+', encoding='utf-8') as file:
                    for p in cluster.newpoints:
                        # we fix the first line of a topic content as the
                        # position to store the top-k keywords, event cluster and event tag
                        file.write(' \n')

                        file.write(p.text)
                        pass
                    pass
                os.rename(old_file, new_file)

                # record new updated topic, and then we'll cal keywords
                if new_file not in self.new_updated_topic:
                    # self.new_updated_topic.append('cluster_{}_{}.txt'.format(cluster.num_points, cluster.id))
                    keyword.update('cluster_{}_{}.txt'.format(cluster.num_points, cluster.id))
                    pass

                # empty topic's new add points
                cluster.newpoints = []

            # if topic not update more than LIFE_CYCLE days, we'll move it to history_dir
            if seg_id - cluster.last_seg_id > self.seg_t:
                # create temp folder in the output path will cause error,
                # so we create temp folder in the parent dir and move the history topic into it

                shutil.move(new_file, os.path.join(self.history_dir,
                                                   'cluster_{}_{}.txt'.format(cluster.num_points, cluster.id)))
                pass
            else:
                new_cluster_list.append(cluster)
                pass
        # maybe some topics have been removed because it is out of LIFE_CYCLE, so we update the previous_topics list
        self.previous_topics = new_cluster_list
        # return self.new_updated_topic