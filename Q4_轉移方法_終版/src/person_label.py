# %% 目標
# 幫每個客戶每年標記轉移的標籤
#
# ```mermaid
# 全體 -> 有轉移(label=2, link=f'{ref}->{landing}')
# 全體 -> 沒轉移
# 沒轉移 -> 轉移到同通路(label=1, link=f'{ref}->{landing=ref}')
# 沒轉移 -> 流失(label=0, link=f'{ref}->loss')
# ```
#
# 所謂轉移:
# 同個{人+標的物}，{明年}有{今年}沒有的通路 & {明年}沒有{今年}有的通路，就算轉換
# Instance 1
# {小明的車子甲}，今年用過A, B, C三種通路:
# 1. 如果明年{小明的車子甲}用的通路是A, D，不算轉換
# 2. 如果明年{小明的車子甲}用的通路是D, E，算轉換
#
# Instance 2
# {小明}有{甲}、{乙}兩台車，今年使用的通路為: 甲-A; 乙-B
# 1. 明年 甲-B，乙-A，這樣算轉換喔。因為 {小明的車子甲}從A -> B，{小明的車子乙}亦然

# %% 使用介面
#
# 1. 讀取每個客戶每個月使用過的通路(在特定險種下)
# ---以下逐年---
# 2. 依規則(今年->明年 or 去年->今年)決定今年的link
# 3. 依上述規則貼label: Literal[0, 1, 2] ; 0=loss, 1=landing==ref, 2=ref!=landing


# %% import
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, get_args, List, Optional, Union
import networkx as nx
import tqdm

ROUTES = ['KA', 'KB', 'BA', 'JB', 'CA', 'Oth']

# %% 讀取資料
def load_data(cate: List[Literal['car_policy_full', 'fir_policy_full', 'hth_policy_full']]=['car_policy_full'],
              path = "D:\\新安\\data\\訂單整合資料"):

    _ROUTES = ['KA', 'KB', 'CA', 'JB', 'BA']

    raw_holder = [pd.read_csv(path+'\\'+c+'.txt', sep='|', dtype=str) for c in cate]
    raw_data = pd.concat(raw_holder)
    # columns typing:
    raw_data[['birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year']] = raw_data[[
        'birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year']].astype(float)
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data["cur_age"] = raw_data["year"] - raw_data["birthY"]
    raw_data["iply_person"] = raw_data["ipolicy"] + raw_data["id"]
    raw_data['iroute'] = raw_data['iroute'].apply(lambda x: 'Oth' if x not in _ROUTES else x)

    used_data = raw_data.query("""
        date >= '2017-01-01' & \
        plyAmt >= 0 & \
        date <= '2022-03-31' & \
        cate != 'H_CV' & \
        cur_age > 0
        """)
    return used_data

# %% 讀取usedRoute或者重新建一個usedRoute
CateOptions = Literal['CAR', 'CA_PS', 'Cf_PS', 'F_PS', 'H_PS', 'H_PS', 'H_PS']
ByOtions = Literal['id', 'target']
def load_usedRoute(raw_data: Optional[pd.DataFrame] = None,
                   cate: CateOptions = 'CAR',
                   by: Union[ByOtions, List[ByOtions]] = 'id',
                   init: bool = True,
                   path: str = "D:\\新安\\A&H 二周目\\Q4_轉移方法_終版\\data\\usedRoute\\",
                   save: bool = True,
                   route_seq: Literal['series', 'unique'] = 'series'
                   ):
    """
    cate: Literal['']
    如果init=False, 讀取；不然重新製作
    """
    if cate not in get_args(CateOptions):
        raise ValueError("cate should be either {opt}".format(opt=get_args(CateOptions)))
    if isinstance(by, str) and by not in get_args(ByOtions):
        raise ValueError("by (str) should be either {opt}".format(opt=get_args(ByOtions)))
    if isinstance(by, list) and set(by).isdisjoint(get_args(ByOtions)):
        raise ValueError("by (list) should be subset of {opt}".format(opt=get_args(ByOtions)))

    if not init:
        routeSeq_name = 'UQ' if route_seq == 'unique' else 'SER'
        by_name = '-'.join(by) if isinstance(by, list) else by
        ans = pd.read_csv(path+ f"usedRoute_{routeSeq_name}_{cate}_{by_name}.csv", sep='|', index_col=by)
        ans.columns = ans.columns.astype(float)
        return ans

    # load data
    if isinstance(raw_data, pd.DataFrame):
        data = raw_data
    else:
        if cate in ['CAR', 'CA_PS', 'Cf_PS']:
            data = load_data(cate=['car_policy_full'])
        if cate in ['F_PS']:
            data = load_data(cate=['fir_policy_full'])
        if cate in ['H_PS', 'H_PS', 'H_PS']:
            data = load_data(cate=['hth_policy_full'])
    cateList = ['CA_PS', 'Cf_PS'] if cate == 'CAR' else [cate]
    data = data.query("cate.isin(@cateList)")
    print("CHECK: cates in orders: {cs}".format(cs=data['cate'].unique()))
    # pivot
    if isinstance(by, list):
        data.sort_values([*by, 'date'], inplace=True)
    else:
        data.sort_values([by, 'date'], inplace=True)

    if route_seq == 'series':
        ans = data.pivot_table(
            index=by,
            columns='year',
            values='iroute',
            aggfunc=lambda x: ','.join(x)
            )
    if route_seq == 'unique':
        ans = data.pivot_table(
            index=by,
            columns='year',
            values='iroute',
            aggfunc=lambda x: ','.join(x.unique())
            )
    if save:
        routeSeq_name = 'UQ' if route_seq == 'unique' else 'SER'
        by_name = '-'.join(by) if isinstance(by, list) else by
        ans.to_csv(path+ f"usedRoute_{routeSeq_name}_{cate}_{by_name}.csv", sep='|')
# %%
raw_data = load_data(['car_policy_full'])
for c in tqdm.tqdm(get_args(CateOptions)):
    for by in ['id', 'target', ['id', 'target']]:
        t = load_usedRoute(raw_data, cate=c, init=True, by=by, save=True, route_seq='series')
        # by_name = '-'.join(by) if isinstance(by, list) else by
        # t.to_csv("D:\\新安\\A&H 二周目\\Q4_轉移方法_終版\\data\\usedRoute\\"+ f"usedRoute_{c}_{by_name}.csv", sep='|')

# %% 轉換
def personal_label(usedRoute: pd.DataFrame, refer_col: str, landing_col: str):
    """
    根據轉換的定義，比較兩年的通路使用情形(有沒有重複都可以)，給每個客戶不同的標籤:
    * 以下以refer_col=今年, landing_col=明年為例 *

    'new in landing year': 今年是空的，明年landing到某些通路。
    'loss': 今年有通路，明年沒有使用，即流失。
    ** 'transfer': 今年的通路 & 明年的通路彼此互斥(且兩年皆非空值)，即目前定義的轉移(0929)。 **
    'stay': 今年的通路 & 明年的通路有overlapping。

    !!注意!! 'stay' 最後計算流量的時候，只算refer_route==landing_route的links。
    """
    refer = usedRoute[refer_col].str.split(',')
    landing = usedRoute[landing_col].str.split(',')
    labels = pd.Series(index=usedRoute.index, name=refer_col, dtype=str)
    # 兩年都是na: 不用考慮
    # refer是na & landing有值: 新客
    labels.loc[refer.isna() & (~landing.isna())] = 'new in landing year'
    # refer有值 & landing是na: 流失
    labels.loc[(~refer.isna()) & landing.isna()] = 'loss'
    # refer isdisjoint landing : 轉移
    mask_nonNa = (~refer.isna()) & (~landing.isna())
    mask_trans = [set(l).isdisjoint(r) if m else False for r, l, m in zip(refer, landing, mask_nonNa)]
    labels.loc[mask_trans] = 'transfer'
    # refer & landing都有值，但refer overlap landing
    mask_stay = (~pd.Series(mask_trans, index=refer.index)) & mask_nonNa
    labels.loc[mask_stay] = 'stay'

    return labels


@dataclass
class LinkAttr:
    """
    refer->landing的屬性 (因為打算把所有年分都塞進同一個r->l的edge才需要)
    """
    cnt: Optional[int] = None
    ids: Optional[str] = None

@dataclass
class Link:
    refer: str
    landing: str
    label: str
    year: Optional[int] = None
    attr: Optional[LinkAttr] = None

def build_flowchart(usedRoute,
                    nodes=ROUTES,
                    years=range(2017, 2022),
                    refer_base:Literal['this year', 'last year'] = 'last year')->List[Link]:

    LAYER_LABELS = ['new in landing year', 'loss', 'stay', 'transfer']

    def _link_match(usedRoute,
                    base_year:int,
                    refer:str,
                    landing:str,
                    refer_base:Literal['this year', 'last year'] = 'last year')->pd.Series:
        """
        符合refer->landing的為True，反之為False.
        e.g.
        base_year = 2017, refer = KA, landing = JB
        某A的
        2017通路為[KA, KB]，此時mask_r (refer) 為True  (KA in [KA, KB])
        2018通路為[KB, CA]，此時mask_l (landing) 為 False (JB in [KB, CA])
        """
        if refer_base not in ['this year', 'last year']:
            raise ValueError(" refer_base should be either 'this year' or 'next year'.")
        if refer_base == 'last year':
            mask_r = usedRoute[base_year].str.contains(refer, na = False)
            mask_l = usedRoute[base_year + 1].str.contains(landing, na = False)
        else:
            mask_r = usedRoute[base_year - 1].str.contains(refer, na = False)
            mask_l = usedRoute[base_year].str.contains(landing, na = False)
        return mask_r & mask_l

    def _labeled_customers(
                           usedRoute,
                           base_year:int, label:str,
                           refer_base:Literal['this year', 'last year'] = 'last year'):
        """
        從usedRoute中篩選出base_year年時，屬於特定label的客戶，並回傳他們base_year和landing年使用的通路
        """
        if refer_base == 'last year':
            mask_label = usedRoute[f'{base_year}_label']==label
            return usedRoute[mask_label][[base_year, base_year+1]]
        else:
            mask_label = usedRoute[f'{base_year}_label']==label
            return usedRoute[mask_label][[base_year-1, base_year]]

    flow_chart: List[Link]  = []
    for label in layer_labels:
        for base_year in years:
            sub_query = _labeled_customers(usedRoute, base_year, label=label, refer_base='last year')
            for r in nodes:
                for l in nodes:
                    mask = _link_match(sub_query, base_year=base_year, refer=r, landing=l, refer_base='last year')
                    ids = sub_query[mask].index.to_list()
                    if label == 'stay' and r != l:
                        flow_chart.append(Link(refer=r, landing=l, label=label+'_virt', year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids) ))
                    else:
                        flow_chart.append(Link(refer=r, landing=l, label=label, year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids) ))
    return flow_chart

fc_CAPS = build_flowchart(usedRoute_CAPS)
flowchart= fc_CAPS
label_list = Literal['new in landing year', 'loss', 'stay', 'transfer']
def flowchart_filter(flowchart: List[Link],
                     label: label_list,
                     refer: str,
                     landing: str)->List[Link]:
    return [f for f in flowchart if f.label==label and f.refer==refer and f.landing==landing]

flowchart_filter(fc_CAPS, 'transfer', refer=)

data = [{'label': f.label, 'year': f.year, 'refer':f.refer, 'landing':f.landing, 'cnt': f.attr.cnt} for f in fc_CAPS]
t = pd.DataFrame(data)
t.query("cnt>0")
t.query("label=='transfer'").groupby('year')['cnt'].sum()
# Q!!! 接近只有一個通路，但是略少，為什麼!???
# 1. 現在說的是人+標的，卻比人還少??
# 2. 就算是人，也不應該比只有一個通路少
