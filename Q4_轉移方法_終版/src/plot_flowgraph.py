import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as ex
from dataclasses import dataclass, field
from typing import Literal, get_args

CateOpts = Literal['CA_PS', 'Cf_PS', 'CAR'] # 已建立的graph & profile的cates
ByOpts = Literal['id', 'id_target'] # 已建立的graph & profile的by method
ROUTES = ['KA', 'KB', 'BA', 'CA', 'JB']
CMAP = ex.colors.qualitative.Vivid
# ============================================================================ #
# 我需要一func幫我以網路圖的形式畫graph的flow
# 需求:
# 1. nodes_x, nodes_y: 每年ROUTES在圖上的位置
# 2. edges: refer, landing, y, flow
#     flow的屬性會需要一個dataclass來處理:
#     a. 太細變虛線
#     b. 粗細的轉換
#     c. visible的切換
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

def hover_info(name, flow_out:list, flow_in:list):
    body = f"<b>{name}<\b><br>"
    body += "<br>===flow out===<br>"
    if len(flow_out) > 0:
        body += '<br>'.join(flow_out)
    body += "<br>===flow in===<br>"
    if len(flow_in) > 0:
        body += '<br>'.join(flow_in)
    return body

def plot_network(cate: CateOpts = 'CA_PS'):
    graph = load_graph(cate, 'id_target', label_new_yearShift = True)
    # ROUTES
    nodes_ROUTES_x = np.repeat(np.arange(2, 2*graph['year'].nunique() + 1, 2),
                               len(ROUTES))
    nodes_ROUTES_y = np.tile(np.arange(1, len(ROUTES) + 1),
                             len(ROUTES))
    names_ROUTES = [r for y in sorted(graph['year'].unique()) for r in ROUTES]
    nodes_ROUTES_colors = np.tile([CMAP[i] for i in range(len(ROUTES))], graph['year'].nunique())

    hoverInfo_ROUTES = []
    y, r = 2020, 'KB'
    for y in sorted(graph['year'].unique()):
        for r in ROUTES:
            flow_out = graph.query("label != 'stay_virt' & year == @y & refer == @r ")\
                        .sort_values('cnt', ascending = False)[['refer', 'landing', 'cnt']]\
                        .apply(lambda x: f"->{x['landing']}: {x['cnt']: ,.0f}", axis = 1)
            flow_in = pd.concat([graph.query("label == 'new' & year == @y & landing == @r "),
                                 graph.query("label not in ['stay_virt', 'new'] & year == @y - 1 & landing == @r ")])\
                        .sort_values('cnt', ascending = False)[['refer', 'landing', 'cnt']]\
                        .apply(lambda x: f"{x['refer']}->: {x['cnt']: ,.0f}", axis = 1)

            hoverInfo_ROUTES.append(hover_info(r, flow_out, flow_in))

    data = []
    data.append(
        go.Scatter(
            x = nodes_ROUTES_x,
            y = nodes_ROUTES_y,
            marker = {'color': nodes_ROUTES_colors,
                      'size': 15,
                      'line_width': 2
                     },
            text = names_ROUTES,
            meta = hoverInfo_ROUTES,
            textposition = 'top center',
            mode = 'markers + text',
            showlegend = False,
            hovertemplate = '%{meta}'
        )
    )
    # add edges
    YEARS = sorted(graph['year'].unique())
    # tranfers
    sub_graph = graph.query("label == 'transfer' ").copy()
    sub_graph['cnt_PRs'] = sub_graph['cnt'].rank() / len(sub_graph)
    edges_trans_x = []
    edges_trans_y = []
    edges_trans_flow = []
    for y in sorted(graph['year'].unique())[:-1]:
        for r in ROUTES:
            for l in ROUTES:
                query_res = sub_graph.query(" year == @y & refer == @r & landing == @l ")
                if not query_res.empty:
                    flow = query_res['cnt_PRs'].iloc[0]
                    edges_trans_y.append([ROUTES.index(r) + 1, ROUTES.index(l) + 1, None])
                    edges_trans_x.append([(YEARS.index(y) + 1) * 2, (YEARS.index(y) + 2) * 2, None])
                    edges_trans_flow.append(flow)

    for x, y, w in zip(edges_trans_x, edges_trans_y, edges_trans_flow):
        if w >= 0.9:
            data.append(
                go.Scatter(
                    x = x,
                    y = y,
                    line = {'color': 'black', 'width': 2},
                    showlegend = False
                )
            )
        else:
            data.append(
                go.Scatter(
                    x = x,
                    y = y,
                    line = {'color': 'black', 'width': w, 'dash': 'dot'},
                    showlegend = False
                )
            )
    fig = go.Figure(data)
    fig.update_layout(title = {'text': f'{cate} (link只畫transfer)'},
                      xaxis = {'tickmode': 'array',
                               'tickvals': np.arange(2, 2*graph['year'].nunique() + 1, 2),
                               'ticktext': YEARS}
                     )
    return fig

if __name__ == '__main__':
    for cate in get_args(CateOpts):
        fig = plot_network(cate)
        pyo.plot(fig, filename = './img/flow_graph_network/flow_network_{cate}.html'.format(cate = cate),
                 auto_open = False)


    for cate in get_args(CateOpts):
        graph = load_graph(cate, 'id_target', label_new_yearShift = True)
        graph.iloc[:, 1:-1].to_csv('./data/flowgraph/without_ids/graph_withouID_{cate}.csv'.format(cate = cate))
