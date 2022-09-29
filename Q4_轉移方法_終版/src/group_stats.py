# %% [markdown]
# 用新的通路辦別方法來分析通路轉換
#
# old: 第y年有踩過就算
#
# new: 第y年的第一個通路 & 第y-1的最後一個通路

# %%
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Literal, Union, List
import networkx as nx

import plotly.express as ex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CMAP = ex.colors.qualitative.Vivid

CATES = ['CA_PS', 'F_PS', 'H_GP', 'H_PS', 'H_TV']
ROUTES = ['KA', 'KB', 'CA', 'JB', 'BA']
FT_DIST = [
    'age', 'plyAmt_CA_PS', 'plyAmt_F_PS', 'plyAmt_H_GP', 'plyAmt_H_PS', 'plyAmt_H_TV',
    'plyAmtOrder_CA_PS', 'plyAmtOrder_F_PS', 'plyAmtOrder_H_GP', 'plyAmtOrder_H_PS',
    'plyAmtOrder_H_TV', 'carCnt', 'carAge', 'clmCnt_order', 'clmAmt_clmOrder', 'clmRate',
    'seqUse_KA', 'seqUse_CA', 'seqUse_BA', 'seqUse_KB', 'seqUse_JB', 'seqUse_Others', 'prefer_KA',
    'prefer_CA', 'prefer_BA', 'prefer_KB', 'prefer_JB', 'prefer_Others'
]

FT_BASIC = [
    'male_ratio', 'marr_ratio', 'n_people', 'clmPeopleRatio', 'carType_其他', 'carType_大型重機',
    'carType_小客車', 'carType_小貨車', 'carType_機車', 'carType_進口車', 'carType_電動自行車', 'insBought_CA_PS',
    'insBought_F_PS', 'insBought_H_GP', 'insBought_H_PS', 'insBought_H_TV', 'n_order_CA_PS',
    'n_order_F_PS', 'n_order_H_GP', 'n_order_H_PS', 'n_order_H_TV'
]

PLOT_FTS = {
    '族群人數': 'n_people',
    '訂單數(車險)': 'n_order_CA_PS',
    '出險人數占比': 'clmPeopleRatio',
    '年齡': 'age',
    '已婚占比': 'marr_ratio',
    '男性占比': 'male_ratio',
    '每人保費_車險': 'plyAmt_CA_PS',
    '每單保費_車險': 'plyAmtOrder_CA_PS',
    '每人每單出險次數': 'clmCnt_order',
    '每張理賠單金額': 'clmAmt_clmOrder',
    '每人損率': 'clmRate',
    '每人車數': 'carCnt',
    '每人車齡': 'carAge',
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


# %%
def ser_des(ser, used_stats=['mean', '25%', '50%', '75%']):
    ans = ser.describe().loc[used_stats]
    ans = ans.rename(index={'25%': 'Pr25', '50%': 'Pr50', '75%': 'Pr75'})
    return ans


def carType_cnt(df):
    cnt = df.filter(like='carType').sum()
    return cnt / cnt.sum()


def insBought_use(df):
    return (df.filter(like='insBought') > 0).mean()


def plyAmt_byOrder(df, cates: list = CATES):
    holder = []
    for c in CATES:
        plyAmt_order_c = df[f'plyAmt_{c}'] / df[f'n_order_{c}']
        des_c = ser_des(plyAmt_order_c[df[f'plyAmt_{c}'] > 0]).rename(f'plyAmtOrder_{c}')
        holder.append(des_c)
    return pd.DataFrame(holder).T


@dataclass
class A_Group:
    """
    year: current year.
    firstRoute: index=the first used route of every custom; columns=years before 'year'.
    lastRoute: index=the last used route of every custom; columns=years before 'year'.
    referral: previous used route before 'year'.
    landing: used route this 'year'.
    population: IDs of customs matched.
    """
    year: int
    firstRoute: pd.DataFrame
    lastRoute: pd.DataFrame
    raw_profile: pd.DataFrame
    referral: Literal['KA', 'KB', 'CA', 'JB', 'BA', 'Others', 'new', 'back', 'all_source'] = 'all'
    landing: Literal['KA', 'KB', 'CA', 'JB', 'BA', 'Others', 'loss', 'all_source'] = 'all'
    method: Literal['from', 'to'] = 'from'
    population: Optional[pd.Series] = None
    profile: Optional[pd.DataFrame] = None

    def __post_init__(self):
        p_refer, p_landing = self.__population_refer(self.method), self.__population_landing(
            self.method)
        self.population = p_refer[p_refer.isin(p_landing)]
        self.profile = self.raw_profile.query('id.isin(@self.population)')

    def __population_refer(self, method='from'):
        assert method in ['from', 'to'], "'method' must be either 'from' or 'to'."
        fr_years = self.firstRoute[[c for c in self.firstRoute.columns if isinstance(c, float)]]

        if self.referral == 'all_source':
            self.__check(self.year - 1, self.firstRoute, 'firstRoute')
            return self.firstRoute['id']  # 所有客戶
        if self.referral == 'new':
            if self.year == min(fr_years.columns):
                return self.firstRoute['id']  # 沒有更之前的年度了
            mask = fr_years[[c for c in fr_years.columns if c < self.year]].isna().all(axis=1)
            return self.firstRoute['id'][mask]
        if self.referral == 'back':
            # y-2(含)以前有買過的人
            mask_before_y_1 = ~fr_years[[c for c in fr_years.columns if c < self.year - 1
                                         ]].isna().all(axis=1)
            # y-1沒買過的人
            mask_y_1 = fr_years[self.year - 1].isna()
            return self.firstRoute['id'][mask_before_y_1 & mask_y_1]

        if method == 'from':
            mask_refer = fr_years[self.year] == self.referral
        else:
            mask_refer = fr_years[self.year - 1] == self.referral
        return self.firstRoute['id'][mask_refer]

    def __population_landing(self, method='from'):
        assert method in ['from', 'to'], "'method' must be either 'from' or 'to'."
        fr_years = self.lastRoute[[c for c in self.lastRoute.columns if isinstance(c, float)]]

        if self.landing == 'all_source':
            self.__check(self.year, self.lastRoute, 'lastRoute')
            return self.lastRoute['id']  # 所有客戶
        if self.landing == 'loss':
            return self.lastRoute['id'][fr_years[self.year].isna()]

        if method == 'from':
            mask_landing = fr_years[self.year + 1] == self.landing
        else:
            mask_landing = fr_years[self.year] == self.landing
        return self.lastRoute['id'][mask_landing]

    def __check(self, col_to_check, df, df_name):
        if col_to_check not in df.columns:
            raise KeyError(f"'{col_to_check}' not in {df_name}.columns.")


@dataclass
class GroupDecribe:
    group: A_Group
    fts_basic: Optional[pd.Series] = None
    fts_dist: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.fts_basic = self.summary_basic(self.group.profile)
        self.fts_dist = self.summary_dist(self.group.profile, self.group.year)

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
            sub_data[[f'n_order_{c}' for c in CATES]].sum(),
        ])

    def summary_dist(self, sub_data, year):
        # 平均、分配類
        return pd.concat(
            [
                # 客戶年齡
                ser_des(year - sub_data['birthY']).rename('age'),
                # 人均總保費
                sub_data[[f'plyAmt_{c}' for c in CATES]].apply(ser_des),
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
            ],
            axis=1)


def get_feat_yearly(graph: nx.classes.digraph.DiGraph, refer: Optional[str], landing: str,
                    feat: str):
    """
    用landing年的資料做統計。

    Args:
    refer: 去年的第一個使用的通路
    landing: 今年的最後使用通路

    if either refer is None and landing is str(e.g. 'KA'), a node query would be applied. edge query would be applied otherwise.
    conditions:
    1. refer is None, landing = 'KA': 'all_source' to 'KA'.
    2. refer = 'KA', landing is None: 'KA' to 'all_source'. it doesn't make sense.
    3. refer = 'KA', landing = 'JB': 'KA' to 'JB'
    """
    if refer is not None:
        query = [
            graph.edges[e[0], e[1]] for e in graph.edges
            if e[0].startswith(refer) & e[1].startswith(landing)
        ]
    else:
        query = [graph.nodes[n] for n in graph.nodes if n.startswith(landing)]

    if feat in FT_DIST:
        ans = pd.concat([i['dist'][feat].rename(i['year']) for i in query if len(i) > 0], axis=1)
        ans = ans[sorted(ans.columns)]
        return ans

    if feat in FT_BASIC:
        ans = pd.DataFrame([[i['basic'][feat] for i in query if len(i) > 0]],
                           columns=[i['year'] for i in query if len(i) > 0],
                           index=[feat])
        ans = ans[sorted(ans.columns)]
        return ans


def get_feat_fig(graph: nx.classes.digraph.DiGraph,
                 refer: Optional[str],
                 landing: str,
                 feat: str,
                 color: str,
                 group_alias: Optional[str] = None,
                 opacity: float = 1,
                 visible: Union[Literal['legendonly'], bool] = 'legendonly',
                 showarea: bool = True,
                 showlegend: bool = True):
    """
    Args:
    refer: 去年的第一個使用的通路
    landing: 今年的最後使用通路
    """

    assert feat in FT_DIST or feat in FT_BASIC, "'feat' should be in FT_DIST or FT_BASIC."

    data = get_feat_yearly(graph, refer, landing, feat)
    if feat in FT_DIST:
        x, y = data.loc['mean'].index, data.loc['mean']
    if feat in FT_BASIC:
        x, y = data.columns, data.loc[feat]
    figData_line = [
        go.Scatter(
            x=x,
            y=y,
            line={'color': color},
            opacity=opacity,
            mode='lines+markers',
            name=group_alias if group_alias is not None else feat,
            visible=visible,
            showlegend=showlegend,
            legendgroup=group_alias if group_alias is not None else feat,
        )
    ]
    if feat in FT_BASIC:
        return figData_line, None

    figData_area = [
        go.Scatter(
            x=x,
            y=data.loc['Pr75'],
            line={
                'color': color,
                'dash': 'dot'
            },
            opacity=opacity * .5,
            mode='lines+markers',
            name=group_alias + '_Pr75',
            visible=visible if showarea else False,
            showlegend=False,
            legendgroup=group_alias if group_alias is not None else feat,
        ),
        go.Scatter(x=x,
                   y=data.loc['Pr25'],
                   line={
                       'color': color,
                       'dash': 'dot'
                   },
                   opacity=opacity * .5,
                   mode='lines+markers',
                   name=group_alias + '_Pr25',
                   visible=visible if showarea else False,
                   showlegend=False,
                   legendgroup=group_alias if group_alias is not None else feat,
                   fill='tonexty'),
        go.Scatter(
            x=x,
            y=data.loc['Pr50'],
            line={
                'color': color,
                'dash': 'dot'
            },
            opacity=opacity * .8,
            mode='lines+markers',
            name=group_alias + '_Pr50',
            visible=visible,
            showlegend=False,
            legendgroup=group_alias if group_alias is not None else feat,
        )
    ]

    return figData_line, figData_area


@dataclass
class GroupSetting:
    refer: str
    landing: str
    feat: str
    color: str = 'black'
    group_alias: Optional[str] = None
    visible: str = 'legendonly'
    showarea: bool = True
    showlegend: bool = True


def make_groups(node: str,
                feat: str,
                method: Literal['from', 'to'] = 'from',
                routes=ROUTES,
                visible_on: list = ['KA', 'JB'],
                showarea: bool = True,
                showlegend: bool = True):
    assert method in ['from', 'to', 'node'], "'method' should be either 'from' or 'to'."
    if method == 'from':
        iters = [(None, node)] + [(node, l) for l in ROUTES]
    if method == 'to':
        iters = [(None, node)] + [(r, node) for r in ROUTES]
    groups = []
    for ie, e in enumerate(iters):
        visible_base = e[0] if method == 'to' else e[1]  # 如果是看從node是從哪個{refer}來，哪些{refer}要顯示，要看e[0]
        groups.append(
            GroupSetting(e[0],
                         e[1],
                         feat,
                         CMAP[ie],
                         group_alias=f'{e[0]}→{e[1]}' if e[0] is not None else e[1],
                         visible=True if visible_base in visible_on else 'legendonly',
                         showarea=showarea,
                         showlegend=showlegend))
    return groups


def plot_aRow(fig, graph: nx.classes.digraph.DiGraph, groups: List[GroupSetting], row: int = 1):
    maxYs_left, maxYs_right = [], []
    for gp in groups:
        showlegend = True if row == 1 else False
        figData_line, figData_area = get_feat_fig(graph,
                                                  gp.refer,
                                                  gp.landing,
                                                  gp.feat,
                                                  color=gp.color,
                                                  group_alias=gp.group_alias,
                                                  visible=gp.visible,
                                                  showarea=gp.showarea)
        if isinstance(figData_line, list):
            maxYs_left.append(max([max(d.y) for d in figData_line]))
            for d in figData_line:
                if row != 1:
                    d.showlegend = False
                fig.add_trace(d, row=row, col=1)

        if isinstance(figData_area, list):
            maxYs_right.append(max([max(d.y) for d in figData_line]))
            for d in figData_area:
                if row != 1:
                    d.showlegend = False
                fig.add_trace(d, row=row, col=2)

    # 如果最大的group比最小的group大10倍，用log
    if maxYs_left and max(maxYs_left) / min(maxYs_left) >= 10:
        fig.update_yaxes(title_text="log", type="log", row=row, col=1)
    if maxYs_right and max(maxYs_right) / min(maxYs_right) >= 10:
        fig.update_yaxes(title_text="log", type="log", row=row, col=2)


def make_yearlyEdges_plots(graph=nx.classes.digraph.DiGraph,
                           node: str = 'KA',
                           method: Literal['from', 'to', 'node'] = 'from',
                           plot_fts: dict = PLOT_FTS,
                           showarea: bool = True,
                           visible_on=['KA', 'JB']):
    subplot_titles = []
    for ft_alias in plot_fts.keys():
        subplot_titles.extend([ft_alias, ''])

    fig = make_subplots(len(plot_fts), 2, shared_yaxes=True, subplot_titles=subplot_titles)
    for rid, ft in enumerate(plot_fts.values(), start=1):
        plot_aRow(fig,
                  graph,
                  make_groups(node,
                              feat=ft,
                              method=method,
                              showlegend=True if rid == 1 else False,
                              showarea=showarea,
                              visible_on=visible_on),
                  row=rid)

    fig.update_layout(title={
        'text': f'<b>... → {node}</b>' if method == 'to' else f'<b>{node} → ...</b>',
        'x': .5,
        'font': {
            'size': 20
        }
    },
                      height=len(plot_fts) * 150,
                      hovermode='x unified')
    return fig


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
