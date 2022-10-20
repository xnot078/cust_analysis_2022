from sklearn import tree
from sklearn.tree import _tree
import networkx as nx
import pandas as pd
import numpy as np

from typing import Union, Optional, Any, Literal
from dataclasses import dataclass, field

def tree2networkx(a_tree, ft_names=None):
    if ft_names is None:
        ft_names = np.arange(len(a_tree.feature))
    G = nx.DiGraph()
    proba = a_tree.value / a_tree.value.sum(axis=2).reshape(-1, 1, 1)
    for cur_idx, (ft_idx, child_left, child_right, thresh) in enumerate(zip(a_tree.feature, a_tree.children_left, a_tree.children_right, a_tree.threshold)):
        if ft_idx >= 0:
            # print(ft_idx, '|', ft_names[ft_idx], '|', child_left, '|', child_right, '|', thresh)
            if child_left>0:
                G.add_edge(cur_idx, child_left, filter = f'{ft_names[ft_idx]}<={thresh:.2f}')
            if child_right>0:
                G.add_edge(cur_idx, child_right, filter = f'{ft_names[ft_idx]}>{thresh:.2f}')
        else:
            G.add_node(cur_idx, proba = proba[cur_idx])
    return G

# G_tree = G2
def get_paths(G_tree):
    """
    !!注意!!
    labe必須是從0開始的integer
    """
    holder = []
    for leaf in [n for n in G_tree.nodes if G_tree.out_degree(n)==0]:
        # print('proba = ', proba[leaf][0])
        path = nx.shortest_path(G_tree, 0, leaf)
        filter = [G_tree.edges[r, l]['filter'] for r, l in zip(path[:-1], path[1:])]
        # print('\t {filters}'.format(filters=' & '.join(filter)))
        proba = G_tree.nodes[leaf]['proba'][0]
        ans = {'query': ' & '.join(filter)}
        ans.update({c: p for c, p in enumerate(proba)})
        holder.append(ans)
    return pd.DataFrame(holder)


@dataclass
class LeafStats:
    """
    n_query_samples: 千萬要小心，因為是把所有樣本(標有pos & neg)都丟進去tree，
                     所有樣本數不會僅限於refer -> landing。
    """
    # params:
    X: pd.DataFrame
    y: pd.DataFrame
    pred: Any
    query: str
    class_weight: Literal['balanced', 'micro'] = 'micro'
    # return:
    query_samples: Optional[pd.DataFrame] = None
    n_query_samples: Optional[int] = None
    actual_valCnt: Optional[pd.Series] = None
    precision: Optional[float] = None

    def __post_init__(self):
        data = pd.concat([self.X, self.y.rename('label')], axis=1)
        self.query_samples = data.query(self.query)
        print('query,', end = '')
        self.n_query_samples = len(self.query_samples)

        actual = self.query_samples['label']
        print('actual,', end = '')
        self.actual_valCnt = pd.Series(actual).value_counts(normalize=True)
        if self.class_weight == 'micro':
            self.precision = (actual == self.pred).mean()
        else:
            # 當class_weight = 'balanced'，precision要用另一種算法
            pos_recall = (sum(actual)/sum(self.y))
            neg_recall = (sum(actual==0)/sum(self.y==0))
            self.precision = pos_recall / (pos_recall+neg_recall)


if __name__ == '__main__':

    from sklearn import datasets
    import re

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns = [re.sub('[^a-z]', '_', i) for i in iris.feature_names])
    y = iris.target
    model = tree.DecisionTreeClassifier(max_features=2, max_depth=2).fit(X, y)
    ft_names = X.columns

    G2 = tree2networkx(model.tree_, ft_names)
    df_filters = get_paths(G2)

    df_filters

    idx = 0
    row = df_filters.loc[idx]

    query, pred = df_filters.loc[0, 'query'], 0
    lfStats = LeafStats(X, y, pred=pred, query=query)
    print(lfStats.precision, lfStats.actual_valCnt.values, lfStats.n_query_samples)

    y = pd.Series(y)

    query_samples = X.query(query)
    n_query_samples = len(query_samples)
    actual = y[query_samples.index]

    actual_valCnt = pd.Series(actual).value_counts(normalize=True)
    precision = (actual == pred).mean()
    actual = y.loc[query_samples.index]

    len(X)
    sum(actual == 0)
    len(actual)

    actual_valCnt = pd.Series(actual).value_counts(normalize=True)
    precision = (actual == pred).mean()
