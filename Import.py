import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import seaborn as sns; sns.set(style='white')
from mpl_toolkits.mplot3d import Axes3D

from wordcloud import WordCloud
from sklearn.decomposition import PCA, TruncatedSVD
import math
import pickle

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import *
import tokenizers
from sklearn.model_selection import StratifiedKFold

pd.set_option('max_colwidth', 40)

