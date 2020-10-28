# author: yx
# date: 2020/9/2 9:10
import torch as t
from yacs.config import CfgNode as CN

# ----------------------------------------------------------------------------------------------------------------------
# Convention about Training/ Test specific parameters
# ----------------------------------------------------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing,
# the corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# ----------------------------------------------------------------------------------------------------------------------
# Config definition
# ----------------------------------------------------------------------------------------------------------------------
_C = CN()

# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.SOGOU_NEWS_CORPUS = '/home/cetc52/yuanxin/data/news_sohu/corpus_seg.txt'

_C.DATASET.FLOW_DATA_CATALOG = '/home/cetc52/yuanxin/data/news_sohu/corpus_day_seg'
_C.DATASET.FLOW_DATA_CATALOG_FIRE = '/home/cetc52/yuanxin/data/news_sohu/corpus_day_seg_fire'

# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------
_C.SOLVER = CN()

# batch size has the same function as seg unit, we think event's life cycle is about 4 days
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.LIFE_CYCLE = 4
_C.SOLVER.SIM_P = 0.3     # similarity threshold for points cluster in each seg
_C.SOLVER.SIM_I = 0.1       # similarity threshold for issue cluster
_C.SOLVER.SIM_T = 0.1     # similarity threshold for topics cluster between previous topic and topic candidates
_C.SOLVER.MIN_COUNT = 0
_C.SOLVER.MIN_FREQ = 1e-5

# ----------------------------------------------------------------------------------------------------------------------
# Misc options
# ----------------------------------------------------------------------------------------------------------------------
_C.STOPWORDS_FILE = '/home/cetc52/yuanxin/topic_detection_tracking/data/chineseStopWords.txt'
_C.OUTPUT_DIR = '/home/cetc52/yuanxin/topic_detection_tracking/output'
_C.HISTORY_DIR = '/home/cetc52/yuanxin/topic_detection_tracking/history'
_C.ISSUE_DIR = '/home/cetc52/yuanxin/topic_detection_tracking/issue'
