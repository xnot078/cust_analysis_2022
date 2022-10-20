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
        by_name = '_'.join(by) if isinstance(by, list) else by
        load_path = path+ f"usedRoute_{routeSeq_name}_{cate}_{by_name}.csv"
        print(load_path)
        ans = pd.read_csv(load_path, sep='|', index_col=by_name)
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
        # ---- !!! 注意 !!!  ---- #
        # 如果直接用id, target當index去pivot，會出現一台車被不同人保，但歸納在同一個人的統計下的情況
        _by = '_'.join(by)
        data[_by] = data[by[0]] + '_' + data[by[1]]
        data.sort_values([_by, 'date'], inplace=True)
        by = _by
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
        save_path = path+ f"usedRoute_{routeSeq_name}_{cate}_{by_name}.csv"
        print(save_path)
        ans.to_csv(save_path, sep='|')

    return ans


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

"""
loss & new都是空的? 對，因為for r in ROUTES: for l in ROUTES 掃不到isna
usedRoute = usedRoute_byID_Target
label = 'new in landing year'
base_year = 2017
"""
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
            if refer == 'new in landing year':
                mask_r = usedRoute[base_year].isna()
            else:
                mask_r = usedRoute[base_year].str.contains(refer, na = False)

            if landing == 'loss':
                mask_l = usedRoute[base_year + 1].isna()
            else:
                mask_l = usedRoute[base_year + 1].str.contains(landing, na = False)
        else:
            if refer == 'new in landing year':
                mask_r = usedRoute[base_year - 1].isna()
            else:
                mask_r = usedRoute[base_year - 1].str.contains(refer, na = False)

            if landing == 'loss':
                mask_l = usedRoute[base_year].isna()
            else:
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

    flow_chart  = []
    for label in LAYER_LABELS:
        for base_year in years:
            sub_query = _labeled_customers(usedRoute, base_year, label=label, refer_base='last year')
            if label == 'new in landing year':
                for l in nodes:
                    mask = _link_match(sub_query, base_year=base_year, refer='new in landing year', landing=l, refer_base='last year')
                    ids = sub_query[mask].index.to_list()
                    lk = Link(refer='new', landing=l, label=label, year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids))
                    lk.__module__ = __name__
                    flow_chart.append(lk)
            if label == 'loss':
                for r in nodes:
                    mask = _link_match(sub_query, base_year=base_year, refer=r, landing='loss', refer_base='last year')
                    ids = sub_query[mask].index.to_list()
                    lk = Link(refer=r, landing='loss', label=label, year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids))
                    lk.__module__ = __name__
                    flow_chart.append(lk)
            else: # transfer and stay
                for r in nodes:
                    for l in nodes:
                        mask = _link_match(sub_query, base_year=base_year, refer=r, landing=l, refer_base='last year')
                        ids = sub_query[mask].index.to_list()
                        if label == 'stay' and r != l:
                            lk = Link(refer=r, landing=l, label=label+'_virt', year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids))
                            lk.__module__ = __name__
                            flow_chart.append(lk)
                        else:
                            lk = Link(refer=r, landing=l, label=label, year=base_year, attr=LinkAttr(cnt=len(ids), ids=ids))
                            lk.__module__ = __name__
                            flow_chart.append(lk)
    return flow_chart

label_list = Literal['new in landing year', 'loss', 'stay', 'transfer']
def flowchart_filter(flowchart: List[Link],
                     label: label_list,
                     refer: str,
                     landing: str)->List[Link]:
    return [f for f in flowchart if f.label==label and f.refer==refer and f.landing==landing]

# %% test
if __name__ == '__main__':

    raw_data = load_data(cate=['car_policy_full'])

    # %%
    for c in tqdm.tqdm(get_args(CateOptions)):
        for by in ['id', 'target', ['id', 'target']]:
            t = load_usedRoute(raw_data, cate=c, init=True, by=by, save=True, route_seq='series')
            # by_name = '-'.join(by) if isinstance(by, list) else by
            # t.to_csv("D:\\新安\\A&H 二周目\\Q4_轉移方法_終版\\data\\usedRoute\\"+ f"usedRoute_{c}_{by_name}.csv", sep='|')

    raw_data['id_target'] = raw_data['id'] + '_' + raw_data['target']
    t = raw_data.pivot_table(index = 'year', columns = 'cate', values = ['id', 'id_target'], aggfunc = 'nunique')
    raw_data.groupby('year')[['id', 'id_target']].nunique()
    t

    # %%　by ['id', 'target'] 的人數會比 usedData by ['id'] 少
    #------------------------------------------------------------------------#
    # usedData by ['id', 'target'] 的人數會比 usedData by ['id'] 少
    # 是因為target.isna()
    #------------------------------------------------------------------------#
    # target是na的保單比例: 0.013
    raw_data['target'].isna().mean()
    # by ['id', 'target']: "人"數 1,333,217  v.s. by ['id']: 人數: 1,408,398
    usedRoute_CAPS_idTar = load_usedRoute(init=False, cate='CA_PS', by=['id', 'target'], route_seq='series')
    usedRoute_CAPS_idTar['id'] = [i.split('_')[0] for i in usedRoute_CAPS_idTar.index]
    usedRoute_CAPS_idTar['target'] = [i.split('_')[1] for i in usedRoute_CAPS_idTar.index]
    print(usedRoute_CAPS_idTar['id'].nunique())
    usedRoute_CAPS_id = load_usedRoute(init=False, cate='CA_PS', by='id', route_seq='series')
    print(len(set(usedRoute_CAPS_id.index)))
    # 兩者的差是否皆為target == nan ? Yes
    diff = set(usedRoute_CAPS_id.index) - set(usedRoute_CAPS_idTar['id'])
    raw_data.query(" id.isin(@diff) and cate=='CA_PS' ")['target'].isna().all()

    for y in range(2017, 2022):
        usedRoute_CAPS_idTar[f'{y}_label'] = personal_label(usedRoute=usedRoute_CAPS_idTar, refer_col=y, landing_col=y+1)
        usedRoute_CAPS_id[f'{y}_label'] = personal_label(usedRoute=usedRoute_CAPS_id, refer_col=y, landing_col=y+1)

    # ========================================================== #
    # 生成網路
    # ========================================================== #
    for cate in ['CA_PS', 'Cf_PS', 'CAR']:
        for by in ['id', ['id', 'target']]:

            cate, by
            usedRoute = load_usedRoute(init=False, cate=cate, by=by, route_seq='series')
            for y in range(2017, 2022):
                usedRoute[f'{y}_label'] = personal_label(usedRoute=usedRoute, refer_col=y, landing_col=y+1)
            grpah = build_flowchart(usedRoute)
            data = [{'label': f.label, 'year': f.year, 'refer':f.refer, 'landing':f.landing, 'cnt': f.attr.cnt, 'ids': ','.join(f.attr.ids)} for f in grpah]
            t = pd.DataFrame(data)
            by_name = by if isinstance(by, str) else '_'.join(by)
            t.query("cnt>0").to_csv(f'./data/flowgraph/graph_{cate}_{by_name}.csv')

    ids = t.query(" year==2017 & label=='transfer' & refer=='KA' & landing=='JB' ")['ids'].iloc[0].split(',')
    len(ids)
    mask_r = usedRoute[2017].str.contains('KA', na=False)
    mask_l = usedRoute[2018].str.contains('JB', na=False)
    mask_label = usedRoute['2017_label'] == 'transfer'
    usedRoute[mask_r&mask_l&mask_label]


    # 兩有個標的物的人的比例 (10~12%)
    for y in range(2017, 2022):
        t = usedRoute_CAPS_idTar[[y,'id','target']]
        tt = t[~t[y].isna()].groupby('id')['target'].nunique()
        ids = tt[tt>1].index
        print(y, (tt>1).mean())
    # ====================================================================== #
    # flow分析:
    #
    # 原本(簡稱為舊):
    # groupby id | filter by isin
    # 現況(簡稱為新):
    # groupby id+target | filter by isin+nonOverlapping
    #
    # 問題:
    # 舊和新誰多誰少？多或少在哪裡？有可能歸納原因嗎？
    # ====================================================================== #

    # ----------------------------------------------------------------- #
    # 以下討論情境:
    # 1. transfer: 2017 KA -> 2018 JB
    # 2. stay:     2017 KA -> 2018 KA
    # 3. new:      2017 na -> 2018 JB
    # 4. loss:     2017 KA -> 2018 na
    # ----------------------------------------------------------------- #

    # ........................................... #
    # 1. transfer: 2017 KA -> 2018 JB
    # ........................................... #
    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].str.contains('JB', na=False)
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_idTar[2018].str.contains('JB', na=False)
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'transfer'
    new_ids_target = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    new_ids = [i.split('_')[0] for i in new_ids_target]
    print('new ids: ', len(set(new_ids)))
    # 有三種情況: 舊有新沒有，舊沒有舊有、舊新都有
    diff_NvO = set(new_ids) - set(old_ids)
    diff_OvN = set(old_ids) - set(new_ids)
    intersct = set(new_ids).intersection(old_ids)
    print(f'Uq Customers: 舊沒有新有({len(diff_NvO):,}); 舊有新沒有{len(diff_OvN):,}; 舊新都有{len(intersct):,}')
    # 那麼舊有新沒有那些人跑去哪裡了呢?
    def where_are_them(ids:List[str], refer:str, landing:str, refer_year:int):
        """
        看看指定的一群{人}，從refer_year到refer_year+1年的{人+標的物}的label數量
        (transfer, transfer_hit(refer->landing), stay, new, loss)
        """
        t = usedRoute_CAPS_idTar.query("id.isin(@ids)").copy()
        mask_r = t[refer_year].str.contains(refer)
        mask_l = t[refer_year + 1].str.contains(landing)
        mask_transfer = t[f'{refer_year}_label'] == 'transfer'
        t.loc[mask_r & mask_l & mask_transfer, f'{refer_year}_label'] = 'transfer_hit' # OvN時可能是空的
        ans = t.pivot_table(
            index='id',
            columns=f'{refer_year}_label',
            values='target',
            aggfunc='count'
        )
        if 'transfer_hit' not in ans.columns:
            ans['transfer_hit'] = pd.NA
        return ans


    # 舊的那些人改用id+tar之後產生了多少流量?
    impact_points = where_are_them(old_ids, 'KA', 'JB', 2017)
    print(f'{len(old_ids):,}->{impact_points.sum().sum():,.0f}')
    impact_points.sum()
    # 舊有新沒有的那些人，他的拆成id+target之後的落點分析
    hide_seek = where_are_them(diff_OvN, 'KA', 'JB', 2017)
    print((hide_seek>0).sum())

    # ........................................... #
    # 2. stay: 2017 KA -> 2018 KA
    # ........................................... #
    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].str.contains('KA', na=False)
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_idTar[2018].str.contains('KA', na=False)
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'stay'
    new_ids_target = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    new_ids = [i.split('_')[0] for i in new_ids_target]
    print('new ids: ', len(set(new_ids)))
    # 有三種情況: 舊有新沒有，舊沒有舊有、舊新都有
    diff_NvO = set(new_ids) - set(old_ids)
    diff_OvN = set(old_ids) - set(new_ids)
    intersct = set(new_ids).intersection(old_ids)
    print(f'Uq Customers: 舊沒有新有({len(diff_NvO):,}); 舊有新沒有{len(diff_OvN):,}; 舊新都有{len(intersct):,}')
    # 舊的那些人改用id+tar之後產生了多少流量?
    impact_points = where_are_them(old_ids, 'KA', 'JB', 2017)
    print(f'{len(old_ids):,}->{impact_points.sum().sum():,.0f}')
    impact_points.sum()
    # 舊有新沒有的那些人，他的拆成id+target之後的落點分析
    hide_seek = where_are_them(diff_OvN, 'KA', 'KA', 2017)
    print((hide_seek>0).sum())

    # ........................................... #
    # 3. new: 2017 na -> 2018 KA
    # ........................................... #
    # old
    mask_r = usedRoute_CAPS_id[2017].isna()
    mask_l = usedRoute_CAPS_id[2018].str.contains('KA', na=False)
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].isna()
    mask_l = usedRoute_CAPS_idTar[2018].str.contains('KA', na=False)
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'new in landing year'
    new_ids_target = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    new_ids = [i.split('_')[0] for i in new_ids_target]
    print('new ids: ', len(set(new_ids)))
    # 有三種情況: 舊有新沒有，舊沒有舊有、舊新都有
    diff_NvO = set(new_ids) - set(old_ids)
    diff_OvN = set(old_ids) - set(new_ids)
    intersct = set(new_ids).intersection(old_ids)
    print(f'Uq Customers: 舊沒有新有({len(diff_NvO):,}); 舊有新沒有{len(diff_OvN):,}; 舊新都有{len(intersct):,}')

    # 舊的那些人改用id+tar之後產生了多少流量?
    impact_points = where_are_them(old_ids, 'new', 'KA', 2017)
    print(f'{len(old_ids):,}->{impact_points.sum().sum():,.0f}')
    impact_points.sum()
    # 新有舊沒有的那些人，他的拆成id+target之後的落點分析
    hide_seek = where_are_them(diff_NvO, 'new', 'KA', 2017)
    print((hide_seek>0).sum())
    usedRoute_CAPS_idTar.query("id=='000fa2c912b26f82dcd2afd00b6974da502030ee'")


    # ........................................... #
    # 4. loss: 2017 KA -> 2018 na
    # ........................................... #
    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].isna()
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_idTar[2018].isna()
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'loss'
    new_ids_target = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    new_ids = [i.split('_')[0] for i in new_ids_target]
    print('new ids: ', len(set(new_ids)))
    # 有三種情況: 舊有新沒有，舊沒有舊有、舊新都有
    diff_NvO = set(new_ids) - set(old_ids)
    diff_OvN = set(old_ids) - set(new_ids)
    intersct = set(new_ids).intersection(old_ids)
    print(f'Uq Customers: 舊沒有新有({len(diff_NvO):,}); 舊有新沒有{len(diff_OvN):,}; 舊新都有{len(intersct):,}')

    # 舊的那些人改用id+tar之後產生了多少流量?
    impact_points = where_are_them(old_ids, 'KA', 'loss', 2017)
    print(f'{len(old_ids):,}->{impact_points.sum().sum():,.0f}')
    impact_points.sum()
    # 舊有新沒有的那些人，他的拆成id+target之後的落點分析
    hide_seek = where_are_them(diff_NvO, 'KA', 'loss', 2017)
    print((hide_seek>0).sum())
    hide_seek[hide_seek['transfer']>0]
    usedRoute_CAPS_idTar.query("id=='001f21fdafe6df7b3721a47066637596243de324'")





    print()


    # %% flowchart的link count v.s. 舊的isin的link的 count 的差距

    # ====================================================================== #
    # flow: transfer
    # by id; (舊)isin v.s. (新1)isin+nonOverlapping
    # 理論上，舊的會比新的多，因為其實新就是舊加上一層"不能重疊"的mask。
    #
    # (舊)by id & isin v.s. (新)by id+target & isin+nonOverlapping
    # 這個比較難說，可能有以下現象:
    # 1. isin中、isin+nonOverlapping沒中 (不會相反)
    # 2. id沒中、id+target中 (會相反，但前者比較多一點); 特例: target isna
    #
    # note: 上述(1)的量級明顯較多

    # ---------------------------------------------------------------------- #
    # (舊)isin v.s. (新)isin+nonOverlapping:
    # ........................................................ #
    # a.新有，舊沒有:
    #                       不存在
    # ........................................................ #

    # ........................................................ #
    # b.舊有，新沒有(2521人):
    #   "沒有"標的物        KA -> JB
    #   但某個標的物有     ... -> JB
    #   同時某個標的物有   KA  -> ...
    #
    #  b-1. stay (以2017 KA -> 2018 JB為例，佔35%)
    #         車A: KA,JB,JB -> JB
    #  b-2. new + stay  (26%)
    #         車A: KA->KA
    #         車B: new->JB
    #  b-3. loss + new    (15%)
    #         車A: KA->loss
    #         車B: new->JB
    #
    # ........................................................ #
    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].str.contains('JB', na=False)
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_idTar[2018].str.contains('JB', na=False)
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'transfer'
    new_ids = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    print('new ids: ', len(set(new_byID_Target_IDsplit)))

    # 新有 舊沒有
    diff_NvsO = set(new_byID_Target_IDsplit) - set(old_ids)
    print(len(diff_NvsO))

    # 舊有 新沒有
    diff_OvsN = set(old_ids) - set(new_byID_Target_IDsplit)
    print(len(diff_OvsN))

    t_diff = usedRoute_CAPS_idTar.query("id.isin(@diff_OvsN)")[[2017, 2018, '2017_label', 'id', 'target']]
    tt = t_diff.pivot_table(index='id', columns='2017_label', values='target', aggfunc='count')
    tt['label'] = tt.apply(lambda x: '|'.join(x.index[x>0]), axis=1)
    tt['label'].value_counts(normalize=True)
    #　案例
    tt[tt['label']=='stay|transfer']
    t_diff.query("id=='f0c618e9c6c77b0b17b4018552393be03b76064d'")
    usedRoute_CAPS_id.loc['f0c618e9c6c77b0b17b4018552393be03b76064d']


    # ---------------------------------------------------------------------- #
    # (舊)id v.s. (新)id+target:
    # ........................................................ #
    # a. 舊有、新沒有 (208):
    #
    #     某個標的物       KA        -> ...(不是JB)
    #     另一個標的物有  ...(不是KA) -> JB
    #
    #     a-1. stay + transfer: (以2017 KA -> 2018 JB為例，佔58%):　
    #             A: JB->JB
    #             B: KA->JB
    #     a-2. loss + transfer (佔10%): transfer中有...->JB; loss補上->KA
    #             A: KA->loss
    #             B: CA->JB (雖有轉換，但不是KA->JB)
    #     a-3. new + transfer (佔3%): transfer中有KA->...; new補上->JB
    #             A: new->JB
    #             B: KA->CA (雖有轉換，但不是KA->JB)
    # ........................................................ #

    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].str.contains('JB', na=False)
    mask_trans = usedRoute_CAPS_id['2017_label'] == 'transfer'
    old_ids = usedRoute_CAPS_id[mask_r & mask_l & mask_trans].index
    print('old ids: ', len(set(old_ids)))
    # new
    mask_r = usedRoute_CAPS_idTar[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_idTar[2018].str.contains('JB', na=False)
    mask_trans = usedRoute_CAPS_idTar['2017_label'] == 'transfer'
    new_ids = usedRoute_CAPS_idTar[mask_r & mask_l & mask_trans].index
    print('new ids: ', len(set(new_byID_Target_IDsplit)))
    diff_OvsN = set(old_ids) - set(new_byID_Target_IDsplit)
    print(len(diff_OvsN))

    t_diff = usedRoute_CAPS_idTar.query("id.isin(@diff_OvsN)")[[2017, 2018, '2017_label', 'id', 'target']]
    tt = t_diff.pivot_table(index='id', columns='2017_label', values='target', aggfunc='count')
    tt['label'] = tt.apply(lambda x: '|'.join(x.index[x>0]), axis=1)
    tt['label'].value_counts(normalize=True)
    #　案例
    tt[tt['label']=='new in landing year|transfer']
    t_diff.query("id=='2ed618891685ffefbc763c81732314f1a0c15395'")
    usedRoute_CAPS_id.loc['2ed618891685ffefbc763c81732314f1a0c15395']

    # ........................................................ #
    # b. 新有、舊沒有(393):
    #
    #     以下情況同時存在:
    #        某個標的物有      KA  -> JB
    #        某個標的物有      JB  -> ...
    #        (或)某個標的物有  ... -> KA
    #
    #     b-1. stay + transfer: (以2017 KA -> 2018 JB為例，佔58%):　transfer中有KA->...; new補上->JB
    #             A: JB->JB
    #             B: KA ->JB
    #     b-2. loss + transfer (佔15%):
    #             A: JB->loss
    #             B: KA->JB (雖有轉換，但不是KA->JB)
    # ........................................................ #
    diff_NvsO = set(new_byID_Target_IDsplit) - set(old_ids)
    print(len(diff_NvsO))

    t_diff = usedRoute_CAPS_idTar.query("id.isin(@diff_NvsO)")[[2017, 2018, '2017_label', 'id', 'target']]
    tt = t_diff.pivot_table(index='id', columns='2017_label', values='target', aggfunc='count')
    tt['label'] = tt.apply(lambda x: '|'.join(x.index[x>0]), axis=1)
    tt['label'].value_counts(normalize=True)
    #　案例
    tt[tt['label']=='new in landing year|stay|transfer']
    t_diff.query("id=='f76a57e7fa0a2c474c543bc63df3e6e47a90c3ef'")
    usedRoute_CAPS_id.loc['f76a57e7fa0a2c474c543bc63df3e6e47a90c3ef']



    # -------------------------------------------------------- #
    # 現在看起來沒什麼問題，但如果我們討論的old是"只有一個標的物的人"呢??
    # 理論上，此時usedRoute_CAPS_idTar == usedRoute_CAPS_id
    # 但當target isna 時，會不一樣 (比例超低, 0.02%)
    # 所以此時和old真的只會差在overlapping
    # -------------------------------------------------------- #
    uqCnt_targets = usedRoute_CAPS_idTar.groupby('id')['target'].nunique()
    ids = uqCnt_targets.index[uqCnt_targets==1]
    a = usedRoute_CAPS_id.loc[ids]
    b = usedRoute_CAPS_idTar.reset_index().set_index('id').loc[ids]

    t = (a[2017.] == b[2017.])
    a[(~t & ~a[2017.].isna())][2017.]
    (~t & ~a[2017.].isna()).mean()
    b[~t & ~b[2017.].isna()][[2017, 'target']]
    (~t & ~b[2017.].isna()).mean()

    # -------------------------------------------------------- #
    # 現在看起來還是沒什麼問題，但如果我們討論的old是"只有一個標的物 & 只有一種通路"的人呢??
    # 理論上，此時old == new 1 == new 2
    # 但當target isna 時，會不一樣 (比例超級低: 1/1872)
    # -------------------------------------------------------- #
    t = usedRoute_CAPS_id[(usedRoute_CAPS_id[2017.]=='KA') & (usedRoute_CAPS_id[2018.]=='JB')]
    mask_hit_ids = is_any_isinDisjoint(t.index, usedRoute_CAPS_idTar, 'KA', 'JB', 2017.)
    # 沒有一個標的物舊符合轉換條件的人
    failed_ids = mask_hit_ids[~mask_hit_ids].index
    # 這些人當中是不是把標的物一起看就符合了?
    tt = usedRoute_CAPS_idTar.query("id.isin(@failed_ids)").groupby('id').apply(lambda x: ('KA' in set(x[2017.])) and 'JB' in set(x[2018.]))
    # 有一個人不符合: target isna
    raw_data.query(" cate=='CA_PS' & year==2017 & id.isin(@tt[~@tt].index)")[['year', 'id', 'target', 'iroute']]
    usedRoute_CAPS_idTar.query("id.isin(@tt[~@tt].index)")
    # 得證

    # -------------------------------------------------------- #
    #
    # 奇怪的例子:
    #
    # 1. usedRoute_id 有，usedRoute_CAPS_idTar中也有，但明明那一年target是nan
    #     raw_data.query("""
    #         cate=='CA_PS' &\
    #         year==2017 &\
    #         id=='0245d7a95d7bd11da82076276198dd585774ff24'
    #         """)[['year', 'id', 'target', 'iroute']]
    #     usedRoute_CAPS_id.filter(like='0245d7a95d7bd11da82076276198dd585774ff24', axis=0)
    #     usedRoute_CAPS_idTar.filter(like='0245d7a95d7bd11da82076276198dd585774ff24', axis=0)
    # 這是因為這個人在2017後開始有17192f7e23e0185beb52d990761a2c945c4a8188這台車。
    #

    # ======================================================== #
    # flow: new
    # 理論上，flow_new不會有重疊的議題，所以如果新(id_targets)舊(id)不同，那應該是多個標的物造成的
    # 新的理論上會比舊的多:
    # 1. 一個人底下，有多個標的物使用同個landing通路
    # 2. 一個人底下，有標的物在refer_year不是na (以標的物來說是new，但以人來說不是)
    # ======================================================== #

    # old
    mask_r = usedRoute_CAPS_id[2017].isna()
    mask_l = usedRoute_CAPS_id[2018].str.contains('JB', na=False)
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index

    flow_byID_Target = build_flowchart(usedRoute_CAPS_idTar)
    new_byID_Target = flowchart_filter(flow_byID_Target, 'new in landing year', 'new', 'JB')
    new_byID_Target = [i for i in new_byID_Target if i.year==2017][0]
    new_byID_Target_IDsplit = [i.split('_')[0] for i in new_byID_Target.attr.ids]
    print(f'old: {len(set(old_ids))}, new: {len(set(new_byID_Target_IDsplit))}')
    diff_new_byID = set(new_byID_Target_IDsplit)-set(old_ids)
    print(f'diff: {len(diff_new_byID)}')
    t = usedRoute_CAPS_idTar.reset_index().set_index('id').loc[diff_new_byID]
    # 有兩個以上標的物使用JB
    usedJB = t[t[2018].str.contains('JB', na=False)].groupby('id')['target'].nunique()
    (usedJB>=2).mean() # 在2017 new-> 2018 JB, 這樣的人+標的物有61.2%
    # 那麼剩下的人+標的物呢? 是否全部都有任何一個標的是2017 != na??
    notNA = ~usedRoute_CAPS_idTar.query("id.isin(@usedJB[@usedJB<2].index)")[2017].isna()
    notNA_ids = [i.split('_')[0] for i in notNA.index]
    set(notNA_ids) == set(usedJB[usedJB<2].index) # 對，都是


    # ======================================================== #
    # flow: loss
    # 理論上，flow_loss應該和flow_new是一樣的邏輯
    # 新的理論上會比舊的多:
    # 1. 一個人底下，有多個標的物使用同個refer通路
    # 2. 一個人底下，有標的物在landing_year不是na (以標的物來說是loss，但以人來說不是)
    # ======================================================== #

    # old
    mask_r = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_l = usedRoute_CAPS_id[2018].isna()
    old_ids = usedRoute_CAPS_id[mask_r & mask_l].index

    new_byID_Target = flowchart_filter(flow_byID_Target, 'loss', 'KA', 'loss')
    new_byID_Target = [i for i in new_byID_Target if i.year==2017][0]
    new_byID_Target_IDsplit = [i.split('_')[0] for i in new_byID_Target.attr.ids]
    print(f'old: {len(set(old_ids))}, new: {len(set(new_byID_Target_IDsplit))}')
    diff_new_byID = set(new_byID_Target_IDsplit)-set(old_ids)
    print(f'diff: {len(diff_new_byID)}')
    t = usedRoute_CAPS_idTar.reset_index().set_index('id').loc[diff_new_byID]
    # 有兩個以上標的物使用KA
    usedKA = t[t[2017].str.contains('KA', na=False)].groupby('id')['target'].nunique()
    (usedKA>=2).mean() # 在2017 new-> 2018 JB, 這樣的人+標的物有61.2%
    # 那麼剩下的人+標的物呢? 是否全部都有任何一個標的是2018 != na??
    notNA = ~usedRoute_CAPS_idTar.query("id.isin(@usedKA[@usedKA<2].index)")[2018].isna()
    notNA_ids = [i.split('_')[0] for i in notNA.index]
    set(notNA_ids) == set(usedKA[usedKA<2].index) # 對，都是


    """========================================================================="""

    # ====================================================================== #
    #------------------------------------------------------------------------#
    # 可能的原因:
    # 1. 重疊
    # 2. 兩年內兩台車以上，且兩年的車不同（表示兩台車，各有唯一的通路）
    # 3. Target is na
    #
    # note. 這兩年只有一台車，理論上應該不存在？
    #
    #  ** 目標是算出兩者差異的組成百分比 **
    #------------------------------------------------------------------------#

    # old: 只by id | 只要isin就算 (不考慮重疊)
    mask_refer = usedRoute_CAPS_id[2017].str.contains('KA', na=False)
    mask_landing = usedRoute_CAPS_id[2018].str.contains('JB', na=False)
    mask_old = mask_refer & mask_landing
    old_link = usedRoute_CAPS_id[mask_old]
    print('old: KA->JB', len(set(old_link.index)))

    # -------------------------------------------------------- #
    # old v.s. new 1
    # isin-(v.s.)-isin+nonOverlapping
    # 理論上只會差在 "重疊"
    # -------------------------------------------------------- #
    flow_byID = build_flowchart(usedRoute_CAPS_id)
    new_byID = flowchart_filter(flow_byID, 'transfer', 'KA', 'JB')
    new_byID = [i for i in new_byID if i.year==2017][0]
    print('new 1 by id:  KA->JB', len(set(new_byID.attr.ids)))
    diff_new_byID = set(old_link.index) - set(new_byID.attr.ids)

    def is_isin_disjoint(ids, usedRoute_byID, refer, landing, base_year):
        """
        檢查一群ids是否明年,今年不同重疊 & refer,landing isin。簡單的說，就是有沒有符合指定refer->landing的轉換定義。
        """
        t = usedRoute_byID.loc[ids][[base_year, base_year+1]] # 鎖定族群
        mask_isin = t[base_year].str.contains(refer) & t[base_year+1].str.contains(landing) # 篩選出有關refer & landing的族群
        t_set = t.applymap(lambda x: set(x.split(',')) if isinstance(x, str) else set())
        disjoint = t_set.apply(lambda x: x[base_year+1].isdisjoint(x[base_year]), axis=1)
        return mask_isin & disjoint

    mask_hit = is_isin_disjoint(ids, usedRoute_CAPS_id, 'KA', 'JB', 2017.) # 簡單的說的話: 不重疊的
    mask_hit.any() # 完全沒有不重疊的。 old v.s. new 1 驗證完畢

    # -------------------------------------------------------- #
    # old v.s. new 2:
    # id-(v.s)-id_target & isin-(v.s.)-isin+nonOverlapping:
    #
    # 理論上會差在 (我需要一個func列出下面三者的百分比):
    # 1. 重疊
    # 2. 標的物整合在id上 v.s. 分開看，分開看會更容易傾向stay而非transfer
    # 3. target.isna (沒有車牌)

    # note. (2.)就是檢驗是否一位符合is_isin_disjoint的id底下，"完全沒有任何[id+標的物]符合is_isin_disjoint"。
    #           if True，表示這個id把標的物合起來看就符合，但拆開就不符合了。
    # -------------------------------------------------------- #
    flow_byID_Target = build_flowchart(usedRoute_CAPS_idTar)
    new_byID_Target = flowchart_filter(flow_byID_Target, 'transfer', 'KA', 'JB')
    new_byID_Target = [i for i in new_byID_Target if i.year==2017][0]
    new_byID_Target_IDsplit = [i.split('_')[0] for i in new_byID_Target.attr.ids]
    print('old (only ids):', len(old_link.index))
    print('new 2 by id & target:  KA->JB', len(set(new_byID_Target.attr.ids)))
    print('new 2 unique ids: ', len(set(new_byID_Target_IDsplit)))
    diff_new_byID_Target = set(old_link.index) - set(new_byID_Target_IDsplit)
    print('old vs new2 (unique ids): ', len(diff_new_byID_Target))

    usedRoute_CAPS_id.loc['00356941fb76eb8ad2361f46e1f7895f7870e0c6']
    usedRoute_CAPS_idTar.query('id.isin(@diff_new_byID_Target)')


    ids = diff_new_byID_Target
    # 重疊的人+標的物
    mask_hit = is_isin_disjoint(id_targets, usedRoute_CAPS_idTar, 'KA', 'JB', 2017.) # 每個id_target是否有符合轉移條件
    mask_hit.any()

    def is_any_isinDisjoint(ids, usedRoute_byID_Target, refer, landing, base_year):
        """
        檢查一群ids中的每一位，是否有任何一個標的物符合is_isin_disjoint (表示這個人有"可以代表轉換"的"單個標的物")
        """
        mask = pd.Series(index=set(ids), dtype=bool)
        usedRoute_byID_Target['id'] = [i.split('_')[0] for i in usedRoute_byID_Target.index]
        usedRoute_byID_Target['target'] = [i.split('_')[1] for i in usedRoute_byID_Target.index]
        id_targets = usedRoute_byID_Target.query("id.isin(@ids)").index
        t = usedRoute_byID_Target.loc[id_targets]
        mask_hit = is_isin_disjoint(id_targets, t, refer, landing, base_year) # 每個id_target是否有符合轉移條件
        mask.loc[t[mask_hit]['id'].unique()] = True # 任一有符合轉移條件標的物的人，視為True
        return mask

    mask_hit_ids = is_any_isinDisjoint(ids, usedRoute_CAPS_idTar, 'KA', 'JB', 2017.)
    mask_hit_ids.any() # 沒有任何人有符合轉移條件的標的物


    """=========================================================================="""
