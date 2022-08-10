import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List
from collections import defaultdict
import datetime, tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

#
paths = {
        "car": "./data_fromChris/car_policy_full.txt",
        "fir": "./data_fromChris/fir_policy_full.txt",
        "hth": "./data_fromChris/hth_policy_full.txt"
        }
raw_data = {k: pd.read_csv(p, sep="|", dtype=str) for k, p in paths.items()}
raw_data = pd.concat(raw_data.values(), axis=0)

# numerical columns:
raw_data[['birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec']] = raw_data[['birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec']].astype(float)
raw_data["date"] = pd.to_datetime(raw_data["date"])
used_data = raw_data.query("""
                        @raw_data['sex'].str.len() <= 1 & \
                        plyAmt >= 0 & \
                        date >= '2017-01-01' & \
                        date <= '2022-03-31' & \
                        cate != 'H_CV'
                        """)
# raw_data["ply_id"] = raw_data["ipolicy"] + raw_data["id"]
used_data["cate"].value_counts()
used_data["cate"] = used_data["cate"].str.replace("Cf_PS", "Car").str.replace("CA_PS", "Car")

iroute = pd.read_csv('./code/policy.iroute.csv')
code_iroute = iroute.set_index('code')['name'].to_dict()
used_data['iroute'] = used_data['iroute'].apply(lambda x: code_iroute.get(x, x))
used_data['iroute'] = used_data['iroute'].str.replace("I.*", "金融通路")

used_data["ply_id"] = used_data["ipolicy"] + '_' + used_data["id"]

used_data["month"] = used_data["date"].dt.month

"""
C_PS 車險個人
F_PS 火險個人（住火）
H_PS HTH個傷
H_GP HTH團傷
H_TV HTH旅平
"""
code_iins = {
    "F_PS": "Fir",
    "H_PS": "個傷",
    "H_GP": "團傷",
    "H_TV": "旅平"
    }
used_data["cate"] = used_data["cate"].apply(lambda x: code_iins.get(x, x))

# 指定各險別top10通路
top_routes = defaultdict(set)
for i in used_data["cate"].unique():
    for y in range(2017, 2023):
        t = used_data.query(" cate==@i and year==@y")
        tops = t.groupby("iroute")["ply_id"].nunique()
        top_routes[i].update(tops.sort_values(ascending=False).head(10).index.to_list())

# 開始流程
"""
new = "在此{險種_i}，此客戶{年度_y}第一次出現"
prev_trans = "在此{險種_i}，此客戶{年度_y} & {年度_(y-1)}有出現"
prev_old = "在此{險種_i}，此客戶{年度_y} & {年度_(<y-1)}有出現。(note: 裡面不應該有prev_trans的人)"
sink = "在此{險種_i}，此客戶{年度_y-1}有出現，但{年度y}沒有出現"
"""

"""
每位平均總保單數: 保單數 / 客戶數
每位平均總簽單保費: 總保費 / 客戶數
每位客戶平均一張的保費: 總保費 / 保單數
每位平均總理賠數: 理賠數 / 客戶數
每位平均總理賠金額: 理賠金 / 客戶數
每位平均損率: 理賠金 / 總保費
每位客戶平均一張的理單金額: 理賠金 / 理賠數
"""
# orders = thisYear_trans
def orders_des(orders):
    """
    給一群orders(通常是根據某些條件篩選後)，算他們的統計量
    cust. : 表示"每位客戶"的平均
    order. : 表示"每張保單"的平均
    case. : 表示"每件賠案"的平均
    """
    des = {
            "客戶數": orders["id"].nunique(),
            "出險客戶數": orders.query("clmCnt>0")["id"].nunique(),
            "保單數": orders["ply_id"].nunique(),
            "保費": orders["plyAmt"].sum(),
            "理賠數": orders["clmCnt"].sum(),
            "理賠金": orders["clmAmt"].sum(),
            "cust.保單數":0,
            "cust.保費":0,
            "cust.理賠件":0,
            "cust.理賠金":0,
            "order.保費":0,
            "case.理賠金":0,
            "損率":0
            }
    if des["客戶數"]>0:
        des["cust.保單數"] = des["保單數"] / des["客戶數"]
        des["cust.保費"] = des["保費"] / des["客戶數"]
        des["cust.理賠件"] = des["理賠數"] / des["客戶數"]
        des["cust.理賠金"] = des["理賠金"] / des["客戶數"]
    if des["出險客戶數"]>0:
        des["cust.理賠金(出險)"] = des["理賠金"] / des["出險客戶數"]
    if des["保單數"]>0:
        des["order.保費"] = des["保費"] / des["保單數"]
    if des["理賠數"]>0:
        des["case.理賠金"] = des["理賠金"] / des["理賠數"]
    if des["保費"]>0:
        des["損率"] = des["理賠金"] / des["保費"]
    return des


# population_des_holder = []
trans_des_holder = []
for i in tqdm.tqdm(used_data["cate"].unique()):
    used_routes = list(top_routes[i])+["其他"]
    for y in range(2017, 2023):

        thisYear = used_data.query("""
                        year==@y & \
                        cate==@i
                        """)
        prevYear = used_data.query("""
                        year==@y-1 & \
                        cate==@i
                        """)
        long_time_ago = used_data.query("""
                        year<@y-1 & \
                        cate==@i
                        """)

        new = thisYear.query("""
                            id not in @prevYear['id'] & \
                            id not in @long_time_ago['id']
                            """)
        prev_old = thisYear.query("""
                            id not in @prevYear['id'] & \
                            id in @long_time_ago['id']
                            """)
        sink = prevYear.query("""
                            id not in @thisYear['id']
                            """)

        # 新客:
        for r_dst in used_routes:
            iroute_exp = "iroute == @r_dst" if r_dst != "其他" else "iroute not in @used_routes"
            g = new.query(f"{iroute_exp}")
            des = orders_des(g)
            des.update({"src": "新客", "dst":r_dst, "tag": "新客", "iins":i, "year_dst":y})
            trans_des_holder.append(des)
        # 舊客_曾買:
        for r_dst in used_routes:
            iroute_exp = "iroute == @r_dst" if r_dst != "其他" else "iroute not in @used_routes"
            g = prev_old.query(f"{iroute_exp}")
            des = orders_des(g)
            des.update({"src": "舊客_曾買", "dst":r_dst, "tag": "舊客_曾買", "iins":i, "year_dst":y})
            trans_des_holder.append(des)
        # 流失:
        for r_src in used_routes:
            iroute_exp = "iroute == @r_src" if r_src != "其他" else "iroute not in @used_routes"
            g = sink.query(f"{iroute_exp}")
            des = orders_des(g)
            des.update({"src": r_src, "dst":"流失", "tag": "流失", "iins":i, "year_dst":y-1})
            trans_des_holder.append(des)
        # 轉移:
        for r_src in used_routes:
            iroute_exp = "iroute == @r_src" if r_src != "其他" else "iroute not in @used_routes"
            prev_src = prevYear.query(f"""
                                    id in @thisYear['id'] & \
                                    {iroute_exp}
                                    """)
            for r_dst in used_routes:
                iroute_exp = "iroute == @r_dst" if r_src != "其他" else "iroute not in @used_routes"
                thisYear_trans = thisYear.query(f"""
                                    id in @prev_src['id'] & \
                                    {iroute_exp}
                                    """)
                des = orders_des(thisYear_trans)
                des.update({"src": r_src, "dst":r_dst, "tag": "轉移", "iins":i, "year_dst":y})
                trans_des_holder.append(des)

des_con = pd.DataFrame(trans_des_holder)
des_con = des_con[['year_dst', 'iins', 'src', 'dst', 'tag',
        '客戶數', '保單數', '保費', '理賠數', '理賠金',
        'cust.保單數', 'cust.保費', 'cust.理賠件',
       'cust.理賠金', 'order.保費', 'case.理賠金', '損率']]
des_con.to_excel("./Q4/Q4_通路轉移_2017-2022.xlsx", index=False)

values = ['客戶數', '保單數', '保費', '理賠數', '理賠金',
        'cust.保單數', 'cust.保費', 'cust.理賠件',
        'cust.理賠金', 'order.保費', 'case.理賠金', '損率']
des_con_melt = []
for _, row in des_con.iterrows():
    for v in values:
        des = {}
        des.update(row.iloc[:5].to_dict())
        des.update({"val_tag": v, "value": row[v]})
        des_con_melt.append(des)
des_con_melt = pd.DataFrame(des_con_melt)
des_con_melt.to_excel("./Q4/Q4_通路轉移_2017-2022_樞紐用.xlsx", index=False)


df = des_old
import numpy as np
def yoy_compare(df, y1=2019, y2=2021, val_tag="客戶數"):
    desPT_year = df.query(f" val_tag=='{val_tag}' ").pivot_table(index=["iins", "dst"], columns="year_dst", values="value")
    pt1 = ((desPT_year[y2]/desPT_year[y1])-1).reset_index().pivot_table(index="iins", columns="dst", values=0)
    pt1 = pt1.applymap(lambda x: x if x<np.inf else np.nan)

    pt2 = ((desPT_year[y2]-desPT_year[y1])).reset_index().pivot_table(index="iins", columns="dst", values=0)
    same_cols = pt1.columns.intersection(pt2.columns)
    sorted_cols = pt2[same_cols].abs().max().sort_values(ascending=False).index

    return pt1[sorted_cols], pt2[sorted_cols]

desPT_year.loc["團傷"]

des_yoy_pt_nor = {}
des_yoy_pt = {}
des_new = des_con_melt.query(" tag=='新客' ")
pts_new = yoy_compare(des_new, y1=2019, y2=2021, val_tag="客戶數")
des_yoy_pt_nor["new"] = pts_new[0]
des_yoy_pt["new"] = pts_new[1]

des_old = des_con_melt.query(" tag=='舊客_曾買' ")
pts_old = yoy_compare(des_old, y1=2019, y2=2021, val_tag="客戶數")
des_yoy_pt_nor["old"] = pts_old[0]
des_yoy_pt["old"] = pts_old[1]

des_reorder = des_con_melt.query(" tag=='轉移' ")
pts_reorder = yoy_compare(des_reorder, y1=2019, y2=2021, val_tag="客戶數")
des_yoy_pt_nor["reorder"] = pts_reorder[0]
des_yoy_pt["reorder"] = pts_reorder[1]


writer = pd.ExcelWriter("./Q4/desPT_YOY.xlsx")
import numpy as np
for i, (k, v) in enumerate(des_yoy_pt.items()):
    print(i, k)
    v.applymap(lambda x: x if x < np.inf else np.nan).to_excel(writer, startrow=(i*7*2)+2
    )
for i, (k, v) in enumerate(des_yoy_pt_nor.items()):
    print(i, k)
    v.applymap(lambda x: x if x < np.inf else np.nan).to_excel(writer, startrow=7+(i*7*2)+2
    )

writer.save()



used_data.query(" cate=='Car' ")["iroute"].value_counts(normalize=True)

top_routes['團傷']
for r_dst in used_routes:
    iroute_exp = "iroute == @r_dst" if r_dst != "其他" else "iroute not in @used_routes"
    g = new.query(f"{iroute_exp}")
    print(r_dst, iroute_exp, len(g))

des_con.query(" src=='新客' ")
des_con_melt.query(" src=='新客' & value>0 & val_tag=='客戶數' ")

top_routes["Car"]
for i in des_con["iins"].unique():
    t = des_con.query("iins==@i")
    for v in values:
        v
t["src"].unique()


pt = t.pivot_table(
                    index="src",
                    columns=["dst", "year_dst"],
                    values=v,
                    aggfunc="sum"
                    )
pt.shape

# 好喔，來驗算看看:

#   1. Car, 2020, 新客->KA
thisYear_KA = used_data.query("""
                cate == 'Car' & \
                year == 2020  & \
                iroute == '員工自行招攬'
                """)
ordersBefore2020 = used_data.query("""
                cate == 'Car' & \
                year < 2020
                """)
newInYear_KA = thisYear_KA.query("""
                id not in @ordersBefore2020['id']
                """)


"""
每位平均總保單數: 保單數 / 客戶數 +
每位平均總簽單保費: 總保費 / 客戶數 +
每位客戶平均一張的保費: 總保費 / 保單數 +
每位平均總理賠數: 理賠數 / 客戶數 +
每位平均總理賠金額: 理賠金 / 客戶數 +
每位平均損率: 理賠金 / 總保費
每位客戶平均一張的理單金額: 理賠金 / 理賠數
"""

        df_src = prev_trans
        df_dst = new
        src_df_mapping = {
                "舊客_曾買": prev_old,
                "舊客_去年": prev_trans,
                "新客": new,
                }

        for src in used_routes+["舊客_曾買", "舊客_去年", "新客"]:
            if src == "舊客_曾買":
                df_src = prev_old
            if src == ""
            for dst in used_routes+["流失"]:



        prevYear
        long_time_ago["year"]
    i
