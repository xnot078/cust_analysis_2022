# %% 目的
# 前情提要:
# 看了group_stats之後，覺得只看平均 & 中位數會不太安心
# 所以想看看dist.
# 需求:
# 1. legend group:
#     refer -> landing 的id_target
#     (僅討論車數的時候，把id_target轉成ids)
# 2. 所有特徵，車數要做特殊處理
# 3. 應該只會用到:
#     (a) ./data/tree_data/positive_idYear
#     (b) ./data/tree_data/profile_all
# 做法:
# 1. 鎖定特定組別的id_target: g
# 2. query (1)的profile
# 3. iter所有特徵: f
# 4. (3)的ser |>　去outliers |> Pr0 : Pr99 的數值
# 5. 把結果存到一個Matix: index = [rerfer->landing, f], columns = 0:100, values = Pr0:Pr99
# 6. Matrix |> Multi-index DataFrame

import pandas as pd
import numpy as np
from typing import Literal, get_args
import re, tqdm

import plotly.express as ex
CMAP = ex.colors.qualitative.Light24
CMAP += ['#ACAFAE'] # ROUTES有五種 => 共計25種組合
import plotly.graph_objs as go
import plotly.offline as pyo

CateOpts = Literal['CA_PS', 'Cf_PS', 'CAR']
Routes = Literal['KA', 'KB', 'CA', 'JB', 'BA']
DefaultShown = ['CA->CA','CA->KA', 'CA->JB',
                 'KA->KA', 'KA->JB']

CNT_FTS = {'carCnt': '每人車數',
           'carType_小客車': '小客車車數',
           'carType_機車': '機車車數',
           'carType_進口車': '進口車車數',
           'carType_小貨車': '小貨車車數',
           'carType_大型重機': '大型重機車數',
           'carType_電動自行車': '電動自行車車數'}

CLM_FTS = {'clmCnt_order': '每(人+車)每單出險次數',
           'clmAmt_clmOrder': '每張理賠單金額',
           'clmRate': '每(人+車)損率'}

PLOT_FTS = { 'cur_age': '年齡',
             'plyAmt_{cate}': '每(人+車)保費_車險',
             'plyAmtOrder_{cate}': '每單保費_車險',

             'car_age': '車齡',
             'seqUse_KA': '最大連續使用_KA',
             'seqUse_CA': '最大連續使用_CA',
             'seqUse_BA': '最大連續使用_BA',
             'seqUse_KB': '最大連續使用_KB',
             'seqUse_JB': '最大連續使用_JB',
             'seqUse_Others': '最大連續使用_Others',
             'prefer_KA': '使用比例_KA',
             'prefer_CA': '使用比例_CA',
             'prefer_BA': '使用比例_BA',
             'prefer_KB': '使用比例_KB',
             'prefer_JB': '使用比例_JB',
             'prefer_Others': '使用比例_Others'}



def load_data(cate:CateOpts = 'CA_PS', cur_year = 2022):
    profile_all = pd.read_parquet('./data/tree_data/profile_all_{cate}.parq'.format(cate = cate))
    profile_all[f'plyAmtOrder_{cate}'] = profile_all[f'plyAmt_{cate}'] / profile_all[f'n_order_{cate}']
    profile_all['cur_age'] = cur_year - profile_all['birthY']
    pos_ids = pd.read_parquet('./data/tree_data/positive_idYear/pos_ids_{cate}.parq'.format(cate = cate))
    return profile_all, pos_ids

def get_group_idTarYear(df_pos_ids, refer: Routes, landing: Routes, return_idTar_only = False):
    """
    輸入refer, landing，從df_pos_ids中query出對應的ids(其實是id_target_year, 簡稱idTY)
    """
    idTY = df_pos_ids.query(" refer == @refer & landing == @landing ")['ids'].iloc[0]
    if return_idTar_only:
        return [re.sub() for i in idTY.split(',')]
    return idTY.split(',')

def get_group_profile(df_profile_all, idTY):
    """
    取得指定ids的prifile
    """
    return df_profile_all.query(" id_year.isin(@idTY) ")

def get_id_carCnt(df_profile_all, ids):
    """
    因現在是by id_target，但希望看到指定的id_target widen到id時(不考慮target)的車輛數(這些人的所有車，不限於refer->landing)
    """
    df_profile_all = df_profile_all.copy()
    df_profile_all['real_id'] = df_profile_all['id'].str.replace('_.*', '', regex = True)
    real_ids = set(re.sub('_.*', '', i) for i in ids)
    df_query = df_profile_all.query("real_id.isin(@real_ids)")

    ans = pd.DataFrame(columns = CNT_FTS.keys(), dtype = float)
    ans['carCnt'] = df_query.groupby('real_id')['target'].nunique()
    for car_ft in CNT_FTS.keys(): # 各車種的不重複車牌數
        if car_ft in df_query.columns:
            ans[car_ft] = df_query[~df_query[car_ft].isna()].groupby('real_id')['target'].nunique()
    return ans

if __name__ == '__main__':
    cate = 'CA_PS'
    profile_all, pos_ids = load_data(cate)
    profile_all.filter(regex = '(?!=.*carType_).*')

def cdf(cate, profile_all, pos_ids):
    PLOT_FTS_CATE = {k.replace('{cate}', cate): v for k, v in PLOT_FTS.copy().items()}
    all_fts = list(PLOT_FTS_CATE.values()) + list(CNT_FTS.values()) + list(CLM_FTS.values())
    all_groups = [f'{r}->{l}' for r in get_args(Routes) for l in get_args(Routes)]
    m_idx = pd.MultiIndex.from_tuples([(ft, r, l)
                              for ft in all_fts
                                for r in get_args(Routes)
                                    for l in get_args(Routes)])
    ft_prs = pd.DataFrame(index = m_idx, columns = np.arange(0, 1, .01))
    ft_mean = pd.DataFrame(index = all_groups, columns = all_fts)
    for r in tqdm.tqdm(get_args(Routes)):
        for l in get_args(Routes):
            ids = get_group_idTarYear(pos_ids, r, l)
            profile_query = get_group_profile(profile_all, ids)

            # feat.s in PLOT_FTS_CATE
            prs = profile_query[profile_query.columns.intersection(PLOT_FTS_CATE.keys())].quantile(np.arange(0, 1, .01))
            prs = prs.rename(columns = PLOT_FTS_CATE).T
            ft_prs.loc[prs.index, r, l] = prs.values

            avg = profile_query[profile_query.columns.intersection(PLOT_FTS_CATE.keys())].mean()
            avg = avg.rename(index = PLOT_FTS_CATE).T
            ft_mean.loc[f'{r}->{l}', avg.index] = avg.values

            # feat.s in CNT_FTS
            car_cnt = get_id_carCnt(profile_all, ids) # 要輸入profile_all 而非profile_query
            prs_carCnt = car_cnt.quantile(np.arange(0, 1, .01))
            prs_carCnt = prs_carCnt.rename(columns = CNT_FTS).T
            ft_prs.loc[prs_carCnt.index, r, l] = prs_carCnt.values

            avg = car_cnt.mean().rename(index = CNT_FTS)
            ft_mean.loc[f'{r}->{l}', avg.index] = avg.values

            # feat.s in CLM_FTS
            prs = profile_query[profile_query.columns.intersection(CLM_FTS.keys())]\
                    .apply(lambda x: x[x>0].quantile(np.arange(0, 1, .01)), axis = 0)
            prs = prs.rename(columns = CLM_FTS).T
            ft_prs.loc[prs.index, r, l] = prs.values

            avg = profile_query[profile_query.columns.intersection(CLM_FTS.keys())]\
                    .apply(lambda x: x[x>0].mean(), axis = 0)\
                    .rename(index = CLM_FTS)
            ft_mean.loc[f'{r}->{l}', avg.index] = avg.values

    # RSE matrix
    links = [f'{r}->{l}' for r in get_args(Routes) for l in get_args(Routes)]
    df_dfm = pd.DataFrame(columns = list(PLOT_FTS.values()) + list(CNT_FTS.values()) + list(CLM_FTS.values()),
                          index = links)

    for ft in set(ft_prs.index.get_level_values(0)):
        df = ft_prs.loc[ft]
        df.index = [f'{x[0]}->{x[1]}' for x in df.index.values]
        # 畫圖前先算個rse
        avg = df.mean().values
        se = np.divide(df-avg, avg, out = np.zeros_like(df), where = avg != 0).mean(axis = 1)
        df_dfm.loc[se.index, ft] = se.values
        # 畫圖
        data = []
        c_id = 0
        data.append(go.Scatter(y = df.columns * 100,
                               x = df.mean().values,
                               line = {'color': 'gray', 'dash': 'dot'},
                               name = '所有分組平均'))
        for name, row in df.iterrows():
            color = CMAP[c_id]
            c_id += 1
            data.append(go.Scatter(y = row.index * 100,
                                   x = row.values,
                                   line = {'color': color},
                                   name = name,
                                   legendgroup = name,
                                   visible = 'legendonly' if name not in DefaultShown else None,
                                   hovertemplate = f'{row.name}.' + '%{x:,.0f}'))
            data.append(go.Scatter(y = [0],
                                   x = [row.mean()],
                                   meta = df_dfm.loc[name, ft],
                                   mode = 'markers',
                                   marker = {'color': color, 'line_width': 1},
                                   name = name,
                                   legendgroup = name,
                                   showlegend = False,
                                   visible = 'legendonly' if name not in DefaultShown else None,
                                   hoverlabel = {'font': {'color': 'black'}},
                                   hovertemplate = f'{row.name}.<br>' + '平均 %{x:,.2f}<br>離均% %{meta:,.2%}'))

        fig = go.Figure(data)
        fig.update_layout(title = {'text': ft, 'font': {'size': 20}, 'x': .5},
                          xaxis = {'title': ft, 'showspikes': True},
                          yaxis = {'title': 'PRs', 'showspikes': True})
        pyo.plot(fig, filename = f'./img/ft_dist/cdf/cdf_{ft}.html', auto_open = False)

    writer = pd.ExcelWriter('./img/ft_dist/cdf/DFM.xlsx')
    df_dfm.to_excel(writer, sheet_name = '離均差')
    df_dfm_rk = df_dfm.rank(ascending=False, axis = 0)
    df_dfm_rk.to_excel(writer, sheet_name = '離均差_Rank')
    pr50 = ft_prs[.5].reset_index()\
              .rename(columns = {'level_0': 'ft',
                                 'level_1': 'refer',
                                 'level_2': 'landing',
                                 .5: 'value'})\
              .assign(link = lambda x: x['refer'] + '->' + x['landing'])\
              .set_index(['ft', 'link'])['value']

    df_dfm_des = pd.concat([df_dfm_rk[ft].map('{:.0f}'.format) +
                             '\n' +
                             df_dfm[ft].map('{:.1%}'.format) +
                             '\n' +
                             ft_mean[ft].map('{:,.0f}'.format) +
                             '\n' +
                             pr50[ft].map('{:,.0f}'.format)
                             for ft in df_dfm_rk], axis = 1)
    df_dfm_des.to_excel(writer, sheet_name = 'des')
    ft_mean.to_excel(writer, sheet_name = '平均')
    writer.save()
    writer.close()

    cdf(profile_all, pos_ids)

    # %% 範例
    import scipy.stats as ss
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    x = np.arange(-3, 5, .1)
    ax.plot(x, ss.norm(2).pdf(x), 'r-.')
    ax.plot(x, ss.norm(2).cdf(x), 'r-')
    # ax.plot(x, ss.norm(0).pdf(x), 'b-.')
    # ax.plot(x, ss.norm(0).cdf(x), 'b-')
    ax.plot(x, ss.gamma.pdf(x, 3, -3), 'b-.')
    ax.plot(x, ss.gamma.cdf(x, 3, -3), 'b-')

    x = np.linspace(ss.norm.ppf(0.01),
                    ss.norm.ppf(0.99), 100)
    ss.norm.cdf(x)
    fig
