# %% import
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Mapping
from collections import defaultdict
import datetime
import tqdm
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

# %% env setting: iroute


@dataclass
class iroute:
    # iroute: code-name mapping
    iroute = pd.read_csv("../data/通路層級.csv")
    iroute = iroute[["iroute", "nroute", "len", "parent", "lv"]]
    iroute_code = iroute.set_index("iroute")["nroute"].dropna().to_dict()
    iroute_code = {k: 'I' if k.startswith(
        'I') else v for k, v in iroute_code.items() if isinstance(k, str)}
    # iroute lv2
    iroute_rules = pd.read_csv("../data/通路第二層規則.csv")
    iroute_rules["lv2 iroute"] = iroute_rules["lv2 iroute"].str.split(',')
    iroute_rules = iroute_rules.set_index("parent")
    iroute_rules = iroute_rules.to_dict()\

    # iroute tops: 各險種的top通路，用來把tops之外的取代成"其他"
    with open('../data/irouteLv1_tops.json', 'r') as f:
        lv1_tops_byCate = json.load(f)
    with open('../data/irouteLv2_tops.json', 'r') as f:
        lv2_tops_byCate = json.load(f)

    def get_lv2(self, iroute_full):
        ir = "I" if iroute_full.startswith("I") else iroute_full[:2]
        if (l := self.iroute_rules["len"].get(ir)) is not None:
            i_lv2 = iroute_full[:int(l)]
            if i_lv2 in self.iroute_rules["lv2 iroute"].get(ir, []):
                return i_lv2
        return None

# %% env setting: env


@dataclass
class env(iroute):
    raw_data_dir: str = r'D:\新安\data\訂單整合資料'
    raw_data: pd.DataFrame = field(init=False)
    filter_query: str = """
        plyAmt >= 0 & \
        date >= '2017-01-01' & \
        date <= '2022-03-31' & \
        cate != 'H_CV' & \
        cate != 'Cf_PS' & \
    	cur_age > 0
        """  # 排除強制險
    used_data: pd.DataFrame = field(init=False)

    def __init__(self):
        self.raw_data_process()
        self.used_data_query()
        self.iroute_mapping()
        self.icarType_mapping()

    def raw_data_process(self):
        # init raw_data
        paths = {
            "car": self.raw_data_dir+"\\car_policy_full.txt",
            "fir": self.raw_data_dir+"\\fir_policy_full.txt",
            "hth": self.raw_data_dir+"\\hth_policy_full.txt"
        }
        # columns typing:
        raw_data = pd.concat(
            {k: pd.read_csv(p, sep="|", dtype=str) for k, p in paths.items()}, axis=0)
        raw_data[['birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year']] = raw_data[[
            'birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year']].astype(float)
        raw_data["date"] = pd.to_datetime(raw_data["date"])
        # new columns:
        raw_data["cur_age"] = raw_data["year"] - raw_data["birthY"]
        raw_data["cur_car_age"] = raw_data["year"] - raw_data["tg_year"]
        raw_data["iply_person"] = raw_data["ipolicy"] + raw_data["id"]
        # iroute:
        self.raw_data = raw_data

    def used_data_query(self):
        self.used_data = self.raw_data.query(self.filter_query)\
                             .assign(
                                 nroute=lambda df: df["iroute"].apply(
                                     lambda x: self.iroute_code.get(x)),
                                 iroute_lv2=lambda df: df["iroute_full"].apply(
                                     self.get_lv2),
        )\
            .assign(
                                 nroute_lv2=lambda df: df["iroute_lv2"].apply(
                                     lambda x: self.iroute_code.get(x))
        )

    def iroute_mapping(self):
        self.used_data["nroute"] = self.used_data[["cate", "nroute"]].apply(
            lambda x: x["nroute"] if x["nroute"] in self.lv1_tops_byCate[x["cate"]].keys() else "其他", axis=1)

        self.used_data["nroute_lv2"] = self.used_data["nroute_lv2"].fillna(
            self.used_data["nroute"])
        self.used_data["nroute_lv2"] = self.used_data[["cate", "nroute_lv2"]].apply(
            lambda x: x["nroute_lv2"] if x["nroute_lv2"] in self.lv2_tops_byCate[x["cate"]].keys() else "其他", axis=1)

    def icarType_mapping(self):
        with open('../data/car_type_tops.json', 'r') as f:
            car_type_tops = json.load(f)
        self.used_data["car_type"] = self.used_data["tg_type"].apply(
            lambda x: car_type_tops.get(x))


# %%agg_functions
"""
要能輸入任一訂單的subset，然後統計出各種特徵
"""


def basic_info(orders):
    if len(orders) < 1:
        return pd.DataFrame()
    return orders.groupby(["year", "id"]).agg(
        female_rate=("sex", lambda g: g.iloc[-1] == '2'),
        age=("cur_age", lambda g: g.iloc[-1]),
        marr=("marr", lambda g: g.iloc[-1] == '2'),
        zip=("zip", lambda g: g.iloc[-1]),
        iply=("ipolicy", "count")
    ).groupby(["year"]).agg(
        female_rate=("female_rate", "mean"),
        age_Avg=("age", "mean"),
        age_Pr25=("age", lambda x: x.quantile(.25)),
        age_Pr50=("age", lambda x: x.quantile(.5)),
        age_Pr75=("age", lambda x: x.quantile(.75)),
        marr_pct=("marr", "mean"),
        zip=("zip", lambda x: x.value_counts(
            normalize=True).head(5).to_dict()),
        n_person=("iply", "count"),
        n_orders=("iply", "sum")
    ).melt(ignore_index=False).reset_index().assign(
        variable_type="basic_info", year_type="this_year"
    )


def iroute_usage(orders):
    """
    通路詳細資訊: ~統計年
    此群體使用過的通路”人次” (一位客戶可能使用過多種通路)
    1.	iroute第一層: 例 保經代90%、網路10%
    2.	iroute第二層 (top5):  例 大成保經10%, …等
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def iroute_usage_lv(sub_orders, on="nroute", name="lv1"):
        ir = sub_orders.groupby(["cate", on])["iply_person"]\
            .nunique()\
            .reset_index()\
            .rename(
            columns={
                f"{on}": "variable",
                "iply_person": "value"
            }
        )\
            .assign(
            year=y,
            year_type="~this_year",
            variable_type="iroute_"+name
        )
        # val -> percentage
        for c in ir["cate"].unique():
            ir.loc[ir["cate"] == c, "value"] = ir.loc[ir["cate"] ==
                                                      c, "value"] / ir.loc[ir["cate"] == c, "value"].sum()

        return ir

    data = []
    for y in sorted(orders["year"].unique()):
        sub_orders = orders.query(" year<= @y ")  # until year_y
        # lv1
        iroute_lv1 = iroute_usage_lv(sub_orders, on="nroute", name="lv1")
        data.append(iroute_lv1)
        # lv2
        iroute_lv2 = iroute_usage_lv(sub_orders, on="nroute_lv2", name="lv2")
        data.append(iroute_lv2)
    return pd.concat(data)


def ins_bought(orders):
    """
    險別資訊(對新安的接觸度): ~統計年、統計年
    ●	此群體接觸過險別_i (例 車險)的百分比。
         (i = [目標的五個險別])
            (買過就算1，不論買了幾張)
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def ins_bought_block(sub_orders, year_type="~this_year"):
        ins = sub_orders.groupby("cate")["id"]\
                        .nunique()\
                        .reset_index()\
                        .rename(
            columns={
                "cate": "variable",
                "id": "value"
            }
        )
        ins["value"] = ins["value"] / \
            sub_orders["id"].nunique()  # value -> percentage
        ins["variable_type"] = "iins_type_bought"
        ins["year_type"] = year_type
        if len(uni_y := sub_orders["year"].unique()) > 1:
            ins["year"] = uni_y.max()
        else:  # otherwise
            ins["year"] = uni_y[0]
        return ins

    data = []
    for y in sorted(orders["year"].unique()):
        ins_b_y = ins_bought_block(orders.query(
            " year==@y "), year_type="this_year")
        data.append(ins_b_y)
        ins_b_until_y = ins_bought_block(
            orders.query(" year<=@y "), year_type="~this_year"
        )
        data.append(ins_b_until_y)
    return pd.concat(data)

# variable	value	variable_type	year_type	year


def target_info_car(orders):
    """
    標的物: 統計年
    ●	車種保單占比 (icar_type)
    ●	車齡: 平均、中位  (ipro_year)
    ●	幾台車：平均、中位  (~統計年)
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def car_type_pct(sub_orders):
        if len(sub_orders) < 1:
            return pd.DataFrame()
        ans = sub_orders["car_type"].fillna("NA")\
            .value_counts(normalize=True)\
            .reset_index()\
            .rename(
            columns={
                "index": "variable",
                "car_type": "value"
            }
        ).assign(
            year=y,
            year_type="this_year",
            variable_type="car_info"
        )
        ans["variable"] = "pct_"+ans["variable"]
        return ans

    def car_age(sub_orders):
        if len(sub_orders) < 1:
            return pd.DataFrame()
        return sub_orders\
            .agg(
                carAge_Avg=("cur_car_age", "mean"),
                carAge_Pr25=("cur_car_age", lambda x: x.quantile(.25)),
                carAge_Pr50=("cur_car_age", lambda x: x.quantile(.5)),
                carAge_Pr75=("cur_car_age", lambda x: x.quantile(.75))
            )\
            .reset_index()\
            .rename(
                columns={
                    "index": "variable",
                    "cur_car_age": "value"
                })\
            .assign(
                year=y, year_type="this_year", variable_type="car_info"
            )

    def car_count(sub_orders):
        if len(sub_orders) < 1:
            return pd.DataFrame()
        no_license = sub_orders.query("target.isna()").groupby("id")[
            "iply_person"].count()
        with_license = sub_orders.groupby("id")["target"].nunique()
        return pd.concat([no_license, with_license], axis=1)\
            .sum(axis=1)\
            .agg(
            carCnt_Avg="mean",
            carCnt_Pr25=(lambda x: x.quantile(.25)),
            carCnt_Pr50=(lambda x: x.quantile(.5)),
            carCnt_Pr75=(lambda x: x.quantile(.75))
        ).reset_index()\
            .rename(
            columns={
                "index": "variable",
                0: "value"
            })\
            .assign(
            year=y, year_type="~this_year", variable_type="car_info"
        )

    data = []
    for y in sorted(orders["year"].unique()):
        c_type = car_type_pct(orders.query(" cate=='CA_PS' & year==@y"))
        c_age = car_age(orders.query(" cate=='CA_PS' & year==@y"))
        c_count = car_count(orders.query(" cate=='CA_PS' & year<=@y"))
        data.extend([c_type, c_age, c_count])
    return pd.concat(data)


def route_loyalty(orders):
    """
    黏著
    連續使用同一通路次數: ~統計年、險別
    此群體險別_i 使用通路_r (第二層)的連續保單數
    ●	第y年與第y-1年的通路不同:
             舊通路的連續保單數
    ●	第y年與第y-1年的通路相同:
             通路的連續保單數
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def iroute_seq(ser, iroute: str):
        init_state = int(iroute in ser)
        state = init_state
        opt = 0
        for i in range(1, len(ser)):
            if ser.iloc[i] == ser.iloc[i-1] == iroute:
                state += 1
            else:
                opt = opt if opt > state else state
                state = init_state
        opt = opt if opt > state else state
        return opt

    def iroute_seq_Series(ser, routes=['KA', 'KB', 'CA', 'JB', 'BA']):
        return pd.Series(
            {f'seqUse_{r}': iroute_seq(ser, r) for r in routes}
        )

    data = []
    for y in sorted(orders["year"].unique()):
        seqRoutes = orders.query(" year<=@y ")\
            .groupby(["cate", "id"])\
            .apply(lambda x: iroute_seq_Series(x["iroute"]))\
            .groupby("cate")\
            .mean()\
            .melt(ignore_index=False)\
            .reset_index()\
            .assign(
            year=y, year_type="~this_year", variable_type="sequential_routes"
        )
        data.append(seqRoutes)

    return pd.concat(data)


def contribution(orders):
    """
    平均每人保費: 統計年、險別
            此群體險別_i的 保費總和 / 人數

    平均客單價: 統計年、險別
            此群體險別_i的 保費總和 / 保單數
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def contri_row(subset):
        ply_cust = subset["plyAmt"].sum() / subset["id"].nunique()

        ply_cust_Pr25, ply_cust_Pr50, ply_cust_Pr75 = subset.groupby(
            "id")["plyAmt"].sum().quantile([.25, .5, .75])
        ply_order = subset["plyAmt"].sum() / subset["iply_person"].nunique()
        ply_order_Pr25, ply_order_Pr50, ply_order_Pr75 = subset["plyAmt"].quantile(
            [.25, .5, .75])

        return pd.Series([ply_cust, ply_cust_Pr25, ply_cust_Pr50, ply_cust_Pr75,
                          ply_order, ply_order_Pr25, ply_order_Pr50, ply_order_Pr75],
                         index=['plyAmt_cust', 'plyAmt_cust_Pr25', 'plyAmt_cust_Pr50', 'plyAmt_cust_Pr75',
                                'plyAmt_order', 'plyAmt_order_Pr25', 'plyAmt_order_Pr50', 'plyAmt_order_Pr75'])

    return orders.groupby(["year", "cate"]).apply(contri_row)\
        .melt(ignore_index=False)\
        .reset_index()\
        .assign(
        year_type="this_year", variable_type="contribution"
    )


def risk(orders):
    """
    賠償人次百分比: 統計年、險別
        有出險過的人次 / 總人次

    損率: 統計年、險別
            此群體在險別_i   賠付總和 / 保費總和

    平均每人賠付金額: 統計年、險別
            此群體在險別_i的 賠付總和 / 有出險的人

    平均一張保單理賠次數: 統計年、險別
            此群體在險別_i的 理賠單  / 有出險的保單數
    """
    if len(orders) < 1:
        return pd.DataFrame()

    def clm_row(subset):
        subset_clm = subset.query("clmAmt>0")
        clm_people = subset_clm["id"].nunique()
        clm_peoplePct = subset_clm["id"].nunique() / subset["id"].nunique()
        clm_rate = subset["clmAmt"].sum() / subset["plyAmt"].sum()
        clm_amtCust = subset["clmAmt"].sum() / subset_clm["id"].nunique()

        clm_amtCust_Pr25, clm_amtCust_Pr50, clm_amtCust_Pr75 = subset_clm.groupby(
            "id")["clmAmt"].sum().quantile([.25, .5, .75])

        clm_cnOrder = subset_clm["clmCnt"].sum(
        ) / subset_clm["iply_person"].nunique()
        clm_cnOrder_Pr25, clm_cnOrder_Pr50, clm_cnOrder_Pr75 = subset_clm["clmCnt"].quantile(
            [.25, .5, .75])
        clm_amtOrder = subset["clmAmt"].sum() / subset_clm["clmCnt"].sum()

        clm_amtOrder_Pr25, clm_amtOrder_Pr50, clm_amtOrder_Pr75 = subset_clm["clmAmt"].quantile(
            [.25, .5, .75])

        return pd.Series(
            [clm_people, clm_peoplePct, clm_rate,
             clm_amtCust, clm_amtCust_Pr25, clm_amtCust_Pr50, clm_amtCust_Pr75,
             clm_amtOrder, clm_amtOrder_Pr25, clm_amtOrder_Pr50, clm_amtOrder_Pr75,
             clm_cnOrder, clm_cnOrder_Pr25, clm_cnOrder_Pr50, clm_cnOrder_Pr75],
            index=["clm_people", "clm_people_pct", "clmRate",
                   "clmAmt_cust", "clm_AmtCust_Pr25", "clm_AmtCust_Pr50", "clm_AmtCust_Pr75",
                   "clmAmt_order", "clm_AmtOrder_Pr25", "clm_AmtOrder_Pr50", "clm_AmtOrder_Pr75",
                   "clmCnt_order", "clm_cntOrder_Pr25", "clm_cntOrder_Pr50", "clm_cntOrder_Pr75"])

    return orders.groupby(["year", "cate"]).apply(clm_row)\
        .melt(ignore_index=False)\
        .reset_index()\
        .assign(
        year_type="this_year", variable_type="contribution"
    )
# %% group_summery func


def group_summary(orders):
    """
    給一個orders，分析當中的客戶，回傳一個melt格式的資料
    """
    return pd.concat([
        basic_info(orders),
        iroute_usage(orders),
        ins_bought(orders),
        target_info_car(orders),
        route_loyalty(orders),
        contribution(orders),
        risk(orders)
    ], axis=0)


# %% init
if __name__ == "__main__":

    e = env()
    used_data = e.used_data

    subset = used_data.sample(5000)

    orders = subset

    # %% 各險種by年度: 新, 續, 回
    filter_query = """
        sex.str.len() <= 1 & \
        plyAmt >= 0 & \
        date >= '2017-01-01' & \
        date <= '2022-03-31' & \
        cate != 'H_CV' & \
        cur_age > 0
        """  # 沒排除強制險

    used_data_tem = e.raw_data.query(filter_query).copy()
    # 車險
    used_data_tem["cate"] = used_data_tem["cate"].str.replace(
        "C.*", "Car", regex=True)

    data_holder = []
    for i in tqdm.tqdm(used_data_tem["cate"].unique()):
        print(f"cate={i}")
        for y in range(2017, 2022):
            print(f"year={y}")
            s = datetime.datetime.today()
            thisYear = used_data_tem.query("""
                        year==@y & \
                        cate==@i
                        """)["id"].astype(str)
            prevYear = used_data_tem.query("""
                            year==@y-1 & \
                            cate==@i
                            """)["id"].astype(str)
            long_time_ago = used_data_tem.query("""
                        year<@y-1 & \
                        cate==@i
                        """)["id"].astype(str)
            new = thisYear[~thisYear.isin(
                prevYear) & ~thisYear.isin(long_time_ago)].nunique()
            trans = thisYear[thisYear.isin(
                prevYear) & ~thisYear.isin(long_time_ago)].nunique()
            prev_old = thisYear[~thisYear.isin(
                prevYear) & thisYear.isin(long_time_ago)].nunique()
            sink = prevYear[~prevYear.isin(thisYear)].nunique()

            data_holder.append(
                pd.Series([i, y, new, trans, prev_old, sink], index=[
                          "cate", "year", "新", "續", "回", "流失"])
            )

    data_holder = pd.DataFrame(data_holder)
    t = used_data_tem.groupby("cate")["id"].nunique()
    t = pd.concat([
        data_holder,
        pd.DataFrame(t).assign(year="不分年").reset_index().rename(
            columns={"id": "總人數"})
    ])
    t.sort_values(["cate", "year"], inplace=True)
    t.to_csv("../data/人數_byYear_Cate.csv", index=False)

    # %% aimming at target1 & target2
    # =============================================================================
    # 今年 v.s. 去年: 續、新
    # 今年 v.s. 去年 & 更久之前: 回流
    # 今年 v.s 明年: 流失
    # =============================================================================
    # population_des_holder = []
    data_holder = []
    for i in tqdm.tqdm(used_data["cate"].unique()):
        print(f"cate={i}")
        for y in range(2017, 2022):
            print(f"year={y}")
            s = datetime.datetime.today()
            thisYear = used_data.query("""
                        year==@y & \
                        cate==@i
                        """)
            prevYear = used_data.query("""
                            year==@y-1 & \
                            cate==@i
                            """)
            # long_time_ago = used_data.query("""
            # 			year<@y-1 & \
            # 			cate==@i
            # 			""")
            # new = used_data.query("""
            # 			id in @thisYear['id'] & \
            # 			id not in @prevYear['id'] & \
            # 			id not in @long_time_ago['id']
            # 			""")
            # prev_old = used_data.query("""
            # 			id in @thisYear['id'] & \
            # 			id not in @prevYear['id'] & \
            # 			id in @long_time_ago['id']
            # 			""")
            # sink = used_data.query("""
            # 			id in @prevYear['id'] & \
            # 			id not in @thisYear['id']
            # 			""")
            # # 新客:
            # print("new...")
            # for dst, g in thisYear.groupby("nroute")["id"]:
            # 	new_subset = new.query("id in @g")
            # 	des = group_summary(new_subset)
            # 	des.assign(
            # 		src = '新客',
            # 		dst = dst
            # 		)
            # 	data_holder.append(des)
            # # 回頭客:
            # print("back...")
            # for dst, g in prev_old.groupby("nroute")["id"]:
            # 	prevOld_subset = prev_old.query("id in @g")
            # 	des = group_summary(prevOld_subset)
            # 	des.assign(
            # 		src = '回頭客',
            # 		dst = dst
            # 		)
            # 	data_holder.append(des)
            # # 流失
            # print("loss...")
            # for src, g in sink.groupby("nroute")["id"]:
            # 	sink_subset = sink.query("id in @g")
            # 	des = group_summary(sink_subset)
            # 	des.assign(
            # 		src = src,
            # 		dst = '流失'
            # 		)
            # 	data_holder.append(des)
            # 轉移:
            print(f"({(datetime.datetime.today()-s).seconds})trans...")
            for src, src_id in prevYear.groupby("nroute")["id"]:
                for dst, dst_id in thisYear.groupby("nroute")["id"]:
                    s = datetime.datetime.today()
                    trans_id = src_id[src_id.isin(dst_id)]
                    trans_subset = used_data.query("id in @trans_id")
                    print(
                        f"\t{src}->{dst}: ({(datetime.datetime.today()-s).seconds})")
                    des = group_summary(trans_subset)
                    des = des.assign(
                        src=src,
                        dst=dst,
                        group_cate=i,
                        group_year=y
                    )
                    data_holder.append(des)

    len(data_holder)
    res = pd.concat(data_holder)
    res = res[['group_year', 'group_cate', 'src', 'dst', 'variable',
               'variable_type', 'year', 'year_type', 'value', 'cate']]
    res.to_csv('../data/feature_trans.csv')
    res.to_excel('../data/feature_trans.xlsx')

    t = e.raw_data.query("""
        sex.str.len() <= 1 & \
        plyAmt >= 0 & \
        date >= '2017-01-01' & \
        date <= '2022-03-31' & \
        cate != 'H_CV'
        """).copy()

    t.reset_index(inplace=True)
    t.loc[t["cate"].str.startswith("C"), "cate"] = "Car"
    t["cate"].unique()
    pt = t.pivot_table(index="year", columns="cate",
                       values="id", aggfunc="nunique")
    ptt = pt.append(t.groupby("cate")["id"].nunique())
    ptt.rename(index={"id": "不分年"}).to_csv("../data/人數_byYear_Cate.csv")
