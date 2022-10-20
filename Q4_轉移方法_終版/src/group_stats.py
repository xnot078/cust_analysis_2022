# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Literal, Union, List, get_args
import networkx as nx
import tqdm, re

import plotly.express as ex
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from sklearn import tree
import src.tree_analyze as ta

CMAP = ex.colors.qualitative.Vivid

CATES = ['CAR', 'Cf_PS', 'CA_PS', 'F_PS', 'H_GP', 'H_PS', 'H_TV']

CateOpts = Literal['CA_PS', 'Cf_PS', 'CAR'] # 已建立的graph & profile的cates
ByOpts = Literal['id', 'id_target'] # 已建立的graph & profile的by method


ROUTES = ['KA', 'KB', 'CA', 'JB', 'BA']
FT_DIST = [
    'age', 'plyAmt_{cate}', 'plyAmt_F_PS', 'plyAmt_H_GP', 'plyAmt_H_PS', 'plyAmt_H_TV',
    'plyAmtOrder_{cate}', 'plyAmtOrder_F_PS', 'plyAmtOrder_H_GP', 'plyAmtOrder_H_PS',
    'plyAmtOrder_H_TV', 'carCnt', 'carAge', 'clmCnt_order', 'clmAmt_clmOrder', 'clmRate',
    'seqUse_KA', 'seqUse_CA', 'seqUse_BA', 'seqUse_KB', 'seqUse_JB', 'seqUse_Others', 'prefer_KA',
    'prefer_CA', 'prefer_BA', 'prefer_KB', 'prefer_JB', 'prefer_Others'
]

FT_BASIC = [
    'male_ratio', 'marr_ratio', 'n_people', 'clmPeopleRatio', 'carType_其他', 'carType_大型重機',
    'carType_小客車', 'carType_小貨車', 'carType_機車', 'carType_進口車', 'carType_電動自行車', 'insBought_{cate}',
    'insBought_F_PS', 'insBought_H_GP', 'insBought_H_PS', 'insBought_H_TV', 'n_order_{cate}',
    'n_order_F_PS', 'n_order_H_GP', 'n_order_H_PS', 'n_order_H_TV'
]

PLOT_FTS = {
    '(人+車)數量': 'n_people',
    '訂單數(車險)': 'n_order_{cate}',
    '出險(人+車)占比': 'clmPeopleRatio',
    '年齡': 'age',
    '已婚占比': 'marr_ratio',
    '男性占比': 'male_ratio',
    '每(人+車)保費_車險': 'plyAmt_{cate}',
    '每單保費_車險': 'plyAmtOrder_{cate}',
    '每(人+車)每單出險次數': 'clmCnt_order',
    '每張理賠單金額': 'clmAmt_clmOrder',
    '每(人+車)損率': 'clmRate',
    # '每(人+車)車數': 'carCnt',
    '車齡': 'carAge',
    '小客車(訂單占比)': 'carType_小客車',
    '機車(訂單占比)': 'carType_機車',
    '進口車(訂單占比)': 'carType_進口車',
    '小貨車(訂單占比)': 'carType_小貨車',
    '大型重機(訂單占比)': 'carType_大型重機',
    '電動自行車(訂單占比)': 'carType_電動自行車',
    '最大連續使用_KA': 'seqUse_KA',
    '最大連續使用_CA': 'seqUse_CA',
    '最大連續使用_BA': 'seqUse_BA',
    '最大連續使用_KB': 'seqUse_KB',
    '最大連續使用_JB': 'seqUse_JB',
    '最大連續使用_Others': 'seqUse_Others',
    '使用比例_KA': 'prefer_KA',
    '使用比例_CA': 'prefer_CA',
    '使用比例_BA': 'prefer_BA',
    '使用比例_KB': 'prefer_KB',
    '使用比例_JB': 'prefer_JB',
    '使用比例_Others': 'prefer_Others'
}


def ser_des(ser, used_stats=['mean', '25%', '50%', '75%']):
    ans = ser.describe().loc[used_stats]
    ans = ans.rename(index={'25%': 'Pr25', '50%': 'Pr50', '75%': 'Pr75'})
    return ans

def carType_cnt(df):
    cnt = df.filter(like='carType').sum()
    return cnt / cnt.sum()

def insBought_use(df):
    return (df.filter(like='insBought') > 0).mean() # 加上target時，即使是車險也可能在今年沒有買而是nan，導致mean < 1

def plyAmt_byOrder(df, cates: list = CATES):
    holder = []
    for c in CATES:
        if f'plyAmt_{c}' in df.columns:
            plyAmt_order_c = df[f'plyAmt_{c}'] / df[f'n_order_{c}']
            des_c = ser_des(plyAmt_order_c[df[f'plyAmt_{c}'] > 0]).rename(f'plyAmtOrder_{c}')
            holder.append(des_c)
    return pd.DataFrame(holder).T



def group_profile(ids, profile=None, cate: Optional[CateOpts]=None, year=None):
    # ============================================================== #
    # 我需要一個func，輸入ids(ids+target)、year、cate(CAR, CA_PS, Cf_PS)
    # 從對應cate, year的profile中，query出ids的subset
    # (如果profile沒有提供，那就去讀取)
    # !! 注意 !!
    # 如果是ids屬於"new in landing year"，year是ids的明年 (year記得+1)
    # ============================================================== #
    if profile is None:
        if cate not in get_args(CateOpts):
            raise ValueError("cate should be either [CA_PS, Cf_PS, CAR].")
        if not (isinstance(year, int) or isinstance(year, float)):
            raise ValueError("year should be a number.")
        profile = pd.read_csv(f'./data/profile/profile_yearly_summary_{cate.replace("_", "")}_{year}.csv')
    return profile.query("id.isin(@ids)")

@dataclass
class GroupDecribe:
    """
    !!注意!!
    如果用id+target，CATE類的只會有對應的車險欄位 (因為target是車牌)
    """
    # params
    profile: pd.DataFrame
    # 以下真的只是用來加標籤用的
    stats_year: int
    refer: str
    landing: str
    label: str
    # return
    stats: Optional[pd.Series] = None

    def __post_init__(self):
        fts_basic = self.summary_basic(self.profile)
        fts_dist = self.summary_dist(self.profile, self.stats_year)

        self.stats = pd.concat([
            fts_dist.melt(ignore_index=False)\
                        .reset_index()\
                        .rename(columns={'index': 'tag', 'variable': 'feature'})\
                        .assign(stats_year = self.stats_year, refer = self.refer, landing = self.landing, label = self.label),
            fts_basic.reset_index()\
                         .rename(columns={'index': 'feature', 0: 'value'})\
                         .assign(tag = None, stats_year = self.stats_year, refer = self.refer, landing = self.landing, label = self.label)
        ]).reset_index(drop=True)


    def summary_basic(self, sub_data):
        # 基本、占比類
        basic = pd.Series(
            [
                # 男性占比
                (sub_data['sex'] == 1).mean(),
                # 已婚占比
                (sub_data['marr'] == 1).mean(),
                # 人數
                len(sub_data),
                # 有理賠的人占比
                (sub_data['clmAmt'] > 0).mean()
            ],
            index=['male_ratio', 'marr_ratio', 'n_people', 'clmPeopleRatio'])

        pct = pd.concat([carType_cnt(sub_data), insBought_use(sub_data)])

        return pd.concat([
            basic,
            pct,
            # 訂單數
            sub_data[[f'n_order_{c}' for c in CATES if f'n_order_{c}' in sub_data.columns]].sum(),
            # sub_data.filter(like='n_order_').sum()
        ])

    def summary_dist(self, sub_data, year):
        # 平均、分配類
        return pd.concat([
                # 客戶年齡
                ser_des(year - sub_data['birthY']).rename('age'),
                # 人均總保費
                sub_data[[f'plyAmt_{c}' for c in CATES if f'plyAmt_{c}' in sub_data.columns]].apply(ser_des),
                # sub_data.filter(like='plyAmt').apply(ser_des),
                # 客單價
                plyAmt_byOrder(sub_data),
                # 車數 & 車齡
                pd.DataFrame([
                    ser_des(sub_data['car_cnt']).rename('carCnt'),
                    ser_des(sub_data['car_age']).rename('carAge')
                ]).T,
                pd.DataFrame([
                    # 有理賠客戶，每位平均一張保單的理賠次數：每位客戶(理賠單數／保單數)
                    ser_des(sub_data.query('clmAmt>0')['clmCnt_order']),
                    # 有理賠客戶，每位平均一張賠案的理賠金額：每位客戶 (理賠金額／理賠單數)
                    ser_des(sub_data.query('clmAmt>0')['clmAmt_clmOrder']),
                    # 有理賠客戶，平均每位的損失率：理賠金額／保費 (有出險的人才算? 對)
                    ser_des(sub_data.query('clmAmt>0')['clmRate'])
                ]).T,
                ser_des(sub_data.filter(like='seq')),
                ser_des(sub_data.filter(like='prefer')),
            ],
            axis=1)


# ============================================================================ #
# 我需要幾個func:
# 1. 輸入指定cate, by id|(id+target)的graph，根據label, year, refer, landing，鎖定flow，並回傳此flow的ids
# 2. 根據指定cate, year讀取特定的profile(或直接給)，並filter出ids
# 3. 計算profile的統計量，並加上cate, label, year, refer, landing資訊
# ============================================================================ #
def load_graph(cate: CateOpts, by: ByOpts, label_new_yearShift = False):
    """
    如果label_new_yearShift = True，label = 'new in landing year'的year往後移一年
    (從refer移到landing year，這樣取ids的時候才會跟其他label一樣)
    """
    df = pd.read_csv('./data/flowgraph/graph_{cate}_{by}.csv'.format(cate = cate, by = by))
    if label_new_yearShift:
        df.loc[df['label']=='new in landing year', 'year'] = df.loc[df['label']=='new in landing year', 'year'] + 1
        df.loc[df['label']=='new in landing year', 'label'] = 'new'
    return df

def get_flow_ids(graph:pd.DataFrame,
                 label: str,
                 year: int,
                 refer: str,
                 landing: str):
    return graph.query("""
                          label==@label &\
                          year==@year &\
                          refer==@refer &\
                          landing==@landing""").iloc[0]['ids'].split(',')

def get_profile(ids,
                profile: Optional[pd.DataFrame] = None,
                cate: Optional[str] = None,
                year: Optional[int] = None,
                return_negative = False):
    if profile is None and\
        cate in get_args(CateOpts) and\
        isinstance(year, int):
        profile = pd.read_csv(f'./data/profile/profile_yearly_summary_{cate.replace("_", "")}_{y}.csv')
    elif profile is None and not (cate in get_args(CateOpts) and isinstance(year, int)):
        raise ValueError("If profile is None, cate and year must be given to load specific profile_file.")

    check_id_col = '_' in profile['id'].iloc[0] # 確認需不需要把id和target合併
    if 'target' in profile.columns and not check_id_col:
        profile['id'] = profile['id'] + '_' + profile['target']

    if return_negative:
        return profile.query("id.isin(@ids)"), profile.query("not id.isin(@ids)")

    return profile.query("id.isin(@ids)")


# ============================================================================ #
# 接著把上述的3個func結合，弄一個輸入cate, return 對應cate的graph的stats
# ============================================================================ #
# cate = 'CA_PS'
def graph_stats(cate: CateOpts = 'CA_PS'):
    graph = load_graph(cate, 'id_target', True)
    holder = []
    for y in range(2017, 2022):
        profile = pd.read_csv(f'./data/profile/profile_yearly_summary_{cate.replace("_", "")}_{y}.csv')
        check_id_col = '_' in profile['id'].iloc[0] # 確認需不需要把id和target合併
        if 'target' in profile.columns and not check_id_col:
            profile['id'] = profile['id'] + '_' + profile['target']
        for r in ROUTES:
            # new
            if y != 2017:
                ids = get_flow_ids(graph, label = 'new', year = y, refer = 'new', landing = r)
                profile_ = get_profile(ids, profile = profile)
                holder.append(
                    GroupDecribe(profile_, label = 'new', refer = 'new', landing = r, stats_year = y).stats
                    )
            # loss
            ids = get_flow_ids(graph, label = 'loss', year = y, refer = r, landing = 'loss')
            profile_ = get_profile(ids, profile = profile)
            holder.append(
                GroupDecribe(profile_, label = 'loss', refer = r, landing = 'loss', stats_year = y).stats
                )
            # stay
            ids = get_flow_ids(graph, label = 'stay', year = y, refer = r, landing = r)
            profile_ = get_profile(ids, profile = profile)
            holder.append(
                GroupDecribe(profile_, label = 'stay', refer = r, landing = r, stats_year = y).stats
                )
            for l in ROUTES:
                if r != l: # transfer
                    ids = get_flow_ids(graph, label = 'transfer', year = y, refer = r, landing = l)
                    profile_ = get_profile(ids, profile = profile)
                    holder.append(
                        GroupDecribe(profile_, label = 'transfer', refer = r, landing = l, stats_year = y).stats
                        )
    return pd.concat(holder)


def graph_pivot(cate: CateOpts = 'CA_PS', save_excel=False):
    des_melt = graph_stats(cate)
    des_melt['flow'] = des_melt['refer'] + '->' + des_melt['landing']
    ft_basic = [i.replace('{cate}', cate) for i in FT_BASIC]
    pt_basic = des_melt.query("feature.isin(@ft_basic)").pivot_table(
        index = ['label', 'flow', 'feature'],
        columns = 'stats_year',
        values = 'value',
        aggfunc = 'mean'
    )
    ft_dist = [i.replace('{cate}', cate) for i in FT_DIST]
    pt_dist = des_melt.query("feature.isin(@ft_dist)").pivot_table(
        index = ['label', 'flow', 'feature'],
        columns = ['tag', 'stats_year'],
        values = 'value',
        aggfunc = 'mean'
    )
    if save_excel:
        writer = pd.ExcelWriter('./data/graph_data/graph_data_{cate}.xlsx'.format(cate = cate))
        des_melt.to_excel(writer, sheet_name='melt data')
        pd.concat([pt_basic, pt_dist['mean']]).to_excel(writer, sheet_name='mean')
        pt_dist['Pr50'].to_excel(writer, sheet_name='Pr50')
        pt_dist['Pr25'].to_excel(writer, sheet_name='Pr25')
        pt_dist['Pr75'].to_excel(writer, sheet_name='Pr75')
        writer.save()
    return namedtuple('graph_data', 'des_melt, pt_basic, pt_dist')(des_melt, pt_basic, pt_dist)

# pivot_table = pt_basic
def plot_aCell(pivot_table,
               feature: str,
               refer:str,
               years = [2017, 2018, 2019, 2020],
               return_fig = False,
               line_dash = 'solid',
               visible_groups = ['JB']):
    t = pivot_table.reset_index().query("flow.str.startswith(@refer) & feature==@feature")
    plot_data = t.set_index('flow')[years].T
    fig_ = ex.line(
        plot_data,
        markers = True
    )
    fig_.update_traces(line={'dash': line_dash})
    for it, t in enumerate(fig_.data):
        r, l = t['name'].split('->')
        if not(r == l or l in visible_groups):
            fig_.data[it].update({'visible': 'legendonly'})

    if return_fig:
        return fig_
    else:
        return fig_.data

# feature = 'male_ratio'
def plot_aRow(cate, pt_basic, pt_dist, feature: str, refer: str):
    # left: mean
    ft_basic = [i.replace('{cate}', cate) for i in FT_BASIC]
    if feature in ft_basic:
        left = plot_aCell(pt_basic, feature, refer, return_fig = False)
        return left, None
    # right: PR50
    left = plot_aCell(pt_dist['mean'], feature, refer, return_fig = False)
    right = plot_aCell(pt_dist['Pr50'], feature, refer, return_fig = False, line_dash='dot')
    return left, right


# profile['seqUse_CA'].mean()
def plot_subplots(pt_basic, pt_dist, cate: CateOpts = 'CA_PS', refer: str = 'KA'):
    """
    args:
        cate: 單純存檔用?
    """
    plot_fts = {k: v.replace('{cate}', cate) for k, v in PLOT_FTS.items()}
    subplot_titles = []
    for ft_alias in plot_fts.keys():
        subplot_titles.extend([ft_alias, ''])
    fig = make_subplots(len(plot_fts), 2, shared_yaxes=True, subplot_titles=subplot_titles)
    for r_idx, (ft_alias, ft) in enumerate(plot_fts.items(), start=1):
        left, right = plot_aRow(cate, pt_basic, pt_dist, feature = ft, refer = refer)
        for t in left:
            if r_idx > 1:
                t.update(showlegend=False) # 避免重複的legend
            fig.add_trace(t, row = r_idx, col = 1)
        if right is not None:
            for t in right:
                t.update(showlegend=False)
                fig.add_trace(t, row = r_idx, col = 2)
    fig.update_layout(title={
                            'text': f'<b>{refer} → ...</b>',
                            'x': .5,
                            'font': {
                                'size': 20
                                }
                            },
                      height=len(plot_fts) * 150,
                      hovermode='x')

    return fig

# ============================================================================ #
# 現在要做決策樹了，
# 1. 輸入ids並視為pos，把剩下的通通視為neg。
# 2. 依據輸入的特徵建立決策樹
# 3. 挑出precision最高的path
# ============================================================================ #


# ============================================================================ #
# 決策樹的輸入:
# 1. refer->landing的不分年pos profile & neg profile。
#     a. refer->landing的id_years for current cate. (made by build_positive_ids())
#     b. profile_all for current cate (made by build_profile_all())
# 2. used_features。
# ============================================================================ #

def build_profile_all(cate: CateOpts = 'CA_PS', return_ans = False):
    """
    把所有年度的profile concat，並加上一個id_year欄位(為了映射指定的refer->landing)
    """
    profile_all = []
    for y in tqdm.trange(2017, 2022):
        profile = pd.read_csv(f'./data/profile/profile_yearly_summary_{cate.replace("_", "")}_{y}.csv')
        check_id_col = '_' in profile['id'].iloc[0] # 確認需不需要把id和`target合併
        if 'target' in profile.columns and not check_id_col:
            profile['id'] = profile['id'] + '_' + profile['target']
        profile['year'] = y
        profile_all.append(profile)
    profile_all = pd.concat(profile_all)
    profile_all['id_year'] = profile_all['id'] + '_' + profile_all['year'].astype(str)
    profile_all.to_parquet(
        './data/tree_data/profile_all_{cate}.parq'.format(cate = cate),
        compression = 'brotli'
        )
    if return_ans:
        return profile_all

def build_positive_ids(cate: CateOpts = 'CA_PS', return_ans = False):
    """
    建立一個DataFrame，包含refer, landing, 和對應的所有id_year
    """
    from collections import defaultdict
    graph = load_graph(cate, 'id_target', True)
    pos_ids = defaultdict(set)
    sub_graph = graph.query(" label != 'stay_virt' ")
    for r in ROUTES:
        for l in ROUTES:
            for _, row in sub_graph.query(" refer == @r & landing == @l ")[['ids', 'year']].iterrows():
                id_year = [j + f'_{row[1]}' for j in row[0].split(',')]
                pos_ids[r, l].update(set(id_year))
    ans = pd.DataFrame([
        [i[0] for i in pos_ids.keys()],
        [i[1] for i in pos_ids.keys()],
        [','.join(i) for i in pos_ids.values()]
        ], index = ['refer', 'landing', 'ids']).T
    ans.to_parquet(
        './data/tree_data/positive_idYear/pos_ids_{cate}.parq'.format(cate = cate),
        compression = 'brotli'
        )
    if return_ans:
        return ans
# build_positive_ids('Cf_PS')
# build_positive_ids('CAR')
# build_profile_all('Cf_PS')
# build_profile_all('CAR')


# used_features = ['sex']
def tree_best_path(pos, neg, used_features: List[str] = ['birthY', 'sex']):
    """
    輸入pos profile & neg profile
    return 一個DecisionTreeClassifier, best_path and its precision (by all)
    """
    if len(pos.columns.intersection(used_features))==0:
        raise ValueError(
                        "used_features not in columns. input: {used_features}".format(
                            used_features = used_features)
                        )
    X = pd.concat([
                   pos[used_features],
                   neg[used_features]
                   ]).fillna(0)
    y = pd.concat([
                   pd.Series(1, index=pos.index, dtype=int),
                   pd.Series(0, index=neg.index, dtype=int)
                   ])
    model = tree.DecisionTreeClassifier(
                    max_depth = len(used_features),
                    max_features = len(used_features),
                    class_weight = 'balanced'
                   ).fit(
                    X, y
                   )
    print('model, ok...', end = '')
    G = ta.tree2networkx(model.tree_, used_features)
    best_path = ta.get_paths(G).sort_values(1, ascending = False).iloc[0]
    best_query, precision = best_path[['query', 1]]
    print('best path, ok...', end = '')
    if best_query:
        best_lf_stats = ta.LeafStats(X, y, pred = 1, query = best_query, class_weight = 'balanced')
        # print(best_lf_stats.precision, best_lf_stats.actual_valCnt.values, best_lf_stats.n_query_samples)
        print('lf_stats, ok...', end = '')
        return best_query, best_lf_stats
    else:
        return False

# ============================================================================ #
# 使用決策樹
# ============================================================================ #
def feature_importance(cate: CateOpts = 'CA_PS',
                       feature_candidates: List[str] = ['sex', 'birthY']):
    """
    為每個一個refer->landing，
    遍歷feature_candidates中的所有features (必須在profile_all.columns中)，
    找出所有特徵的precision(和最佳tree path)
    """
    profile_all = pd.read_parquet('./data/tree_data/profile_all_{cate}.parq'.format(cate = cate))
    pos_ids = pd.read_parquet('./data/tree_data/positive_idYear/pos_ids_{cate}.parq'.format(cate = cate))

    # r, l = 'KA', 'KA'
    # ft = 'sex'
    holder = []
    for r in ROUTES:
        for l in ROUTES:
            id_year = pos_ids.query(" refer == @r & landing == @l")['ids'].iloc[0].split(',')
            pos = profile_all.query('id_year.isin(@id_year)')
            neg = profile_all.query('not id_year.isin(@id_year)')

            print(f'{r}->{l}')
            for ft in feature_candidates:
                print(ft, end = ': ')
                if ft not in pos.columns:
                    print(f'{ft} not in columns.')
                    continue
                best_query, stats = tree_best_path(pos, neg, [ft])
                if tree_best_path != False:
                    # stats.precision, stats.n_query_samples
                    holder.append({'refer': r,
                                   'landing': l,
                                   'best_query': best_query,
                                   'precision': stats.precision,
                                   'n_samples': stats.n_query_samples})
                    print('append, ok')
                else:
                    print('pass')
    return pd.DataFrame(holder)


# %%
used_fts = ['sex', 'marr', 'birthY', 'seqUse_KA',
            'seqUse_CA', 'seqUse_BA', 'seqUse_KB', 'seqUse_JB', 'seqUse_Others',
            'prefer_KA', 'prefer_CA', 'prefer_BA', 'prefer_KB', 'prefer_JB',
            'prefer_Others', 'carType_其他', 'carType_大型重機', 'carType_小客車',
            'carType_小貨車', 'carType_機車', 'carType_進口車', 'carType_電動自行車', 'car_age',
            'car_cnt', 'n_orders', 'n_clm',
            'clmAmt', 'clmCnt', 'clmCnt_order', 'clmAmt_clmOrder', 'clmRate']

for cate in get_args(CateOpts): # ['CAR', 'Cf_PS']:#
    ft_ipt = feature_importance(cate, used_fts+[f'insBought_{cate}', f'plyAmt_{cate}', f'n_order_{cate}'])
    # ft_ipt.sort_values('precision')
    ft_ipt['usability'] = ft_ipt.apply(lambda x: not re.match('.*'+x['refer'], x['best_query']), axis = 1)
    ft_ipt['des'] = ft_ipt['best_query'] + '\n' + ft_ipt['precision'].round(2).astype(str) + '\n' + ft_ipt['n_samples'].map('{:,.0f}'.format)
    ft_ipt.to_csv('./data/tree_data/feature_importance/ft_importance_{cate}.csv'.format(cate = cate), encoding = 'utf-8')

ft_ipt.query("refer == 'KA' & landing == 'JB' & not best_query.str.contains('KA') ").sort_values('precision', ascending = False)

# %%
cate = 'CA_PS'
for cate in get_args(CateOpts):
    _, pt_basic, pt_dist = graph_pivot(cate, save_excel = True)
    for r in ROUTES:
        fig = plot_subplots(pt_basic, pt_dist, cate = cate, refer =r)
        fig.update_layout(hovermode='x')
        pyo.plot(fig,
                 auto_open = False,
                 filename = './img/{cate}/{cate}_{refer}_to.thml'.format(cate = cate, refer = r))

fig
# ============================================================================ #
# 接著是決策樹:
# 輸入postive profile & negative profile >> balance training samples >> train (one ft a time)
# >> find most impacting feature (with the highest precision)
# >> multi-fts
# ============================================================================ #


# %%
if __name__ == "__main__":
    # 指定特定險種的: 每個客戶在年度y的第一個通路 & 最後通路
    CATE = 'CA_PS'
    PATH = r'D:\新安\A&H 二周目\Q4\notebook\特徵群組分析\data\profile'

    firstRoute = pd.read_csv(PATH + '\\firstRoute_year_{c}.csv'.format(c=CATE))
    lastRoute = pd.read_csv(PATH + '\\lastRoute_year_{c}.csv'.format(c=CATE))
    firstRoute.columns = [float(c) if c.startswith('20') else c for c in firstRoute.columns]
    lastRoute.columns = [float(c) if c.startswith('20') else c for c in lastRoute.columns]

    # 建立資料
    refers = [
        'KA',
        'KB',
        'CA',
        'JB',
        'BA',
        'Others',
    ]
    landing = [
        'KA',
        'KB',
        'CA',
        'JB',
        'BA',
        'Others',
    ]
    G_from = nx.DiGraph()
    G_to = nx.DiGraph()
    start_y, end_y = 2018, 2022
    for y in range(start_y, end_y):
        data = pd.read_csv(PATH + '\\profile_yearly_summary_' + str(y) + '.csv')
        #建立 edges 的統計
        for r in refers:
            for l in landing:
                if y > start_y and y <= end_y:
                    a_group = A_Group(y,
                                      firstRoute=firstRoute,
                                      lastRoute=lastRoute,
                                      raw_profile=data,
                                      referral=r,
                                      landing=l,
                                      method='from')
                    group_des = GroupDecribe(a_group)
                    G_from.add_edge(r + f'_{y-1}',
                                    l + f'_{y}',
                                    year=y,
                                    dist=group_des.fts_dist,
                                    basic=group_des.fts_basic)
        for r in refers:
            for l in landing:
                if y >= start_y and y < end_y:
                    a_group = A_Group(y,
                                      firstRoute=firstRoute,
                                      lastRoute=lastRoute,
                                      raw_profile=data,
                                      referral=r,
                                      landing=l,
                                      method='to')
                    group_des = GroupDecribe(a_group)
                    G_to.add_edge(r + f'_{y}',
                                  l + f'_{y+1}',
                                  year=y,
                                  dist=group_des.fts_dist,
                                  basic=group_des.fts_basic)

        # 建立 nodes 的統計
        for l in landing:
            a_group = A_Group(y,
                              firstRoute=firstRoute,
                              lastRoute=lastRoute,
                              raw_profile=data,
                              referral='all_source',
                              landing=l,
                              method='to')  # 如果method='from'，node的變成"明年"的狀態。
            group_des = GroupDecribe(a_group)
            nx.set_node_attributes(
                G_from,
                {l + f'_{y}': dict(year=y, dist=group_des.fts_dist, basic=group_des.fts_basic)})

            nx.set_node_attributes(
                G_to,
                {l + f'_{y}': dict(year=y, dist=group_des.fts_dist, basic=group_des.fts_basic)})

    nx.write_gpickle(G_from, PATH + '\\graph_from.gpickle')
    nx.write_gpickle(G_to, PATH + '\\graph_to.gpickle')

# %%
