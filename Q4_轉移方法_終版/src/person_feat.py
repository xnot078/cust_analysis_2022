# %%
import pandas as pd
import json
# typing
from dataclasses import dataclass, field

# %%
ROUTES = ['KA', 'KB', 'CA', 'JB', 'BA', 'Others']


# %%
@dataclass
class env():
    raw_data_dir: str = r'D:\新安\data\訂單整合資料'
    raw_data: pd.DataFrame = field(init=False)
    filter_query: str = """
        plyAmt >= 0 & \
        date >= '2017-01-01' & \
        date <= '2022-03-31' & \
        cate != 'H_CV' & \
    	cur_age > 0
        """  # 排除強制險cate != 'Cf_PS' & \
    used_data: pd.DataFrame = field(init=False)

    def __init__(self):
        self.raw_data_process()
        self.used_data_query()
        self.icarType_mapping()

    def raw_data_process(self):
        # init raw_data
        paths = {
            "car": self.raw_data_dir + "\\car_policy_full.txt",
            "fir": self.raw_data_dir + "\\fir_policy_full.txt",
            "hth": self.raw_data_dir + "\\hth_policy_full.txt"
        }
        # columns typing:
        raw_data = pd.concat({k: pd.read_csv(p, sep="|", dtype=str)
                              for k, p in paths.items()},
                             axis=0)
        raw_data[['birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year']] = raw_data[[
            'birthY', 'year', 'plyAmt', 'clmCnt', 'clmAmt', 'clmRec', 'tg_year'
        ]].astype(float)
        raw_data["date"] = pd.to_datetime(raw_data["date"])
        # new columns:
        raw_data["cur_age"] = raw_data["year"] - raw_data["birthY"]
        raw_data["cur_car_age"] = raw_data["year"] - raw_data["tg_year"]
        raw_data["iply_person"] = raw_data["ipolicy"] + raw_data["id"]
        # iroute:
        raw_data.loc[~raw_data["iroute"].isin(ROUTES), "iroute"] = 'Others'
        self.raw_data = raw_data

    def used_data_query(self):
        self.used_data = self.raw_data.query(self.filter_query)

    def icarType_mapping(self):
        with open(f'D:\新安\A&H 二周目\data\car_type_tops.json', 'r') as f:
            car_type_tops = json.load(f)
        self.used_data["car_type"] = self.used_data["tg_type"].apply(lambda x: car_type_tops.get(x))


# %%
e = env()

# %%
data = e.used_data

# data.query(" id == '00754e1a3c88875f44d5e5ac02ddb2024adc8cca' & target == '69fd3caf6636d913415aae10a8e6e74cc82ac506' ")


# # %%
# for c in ['CA_PS', 'H_GP', 'H_TV', 'H_PS', 'F_PS']:
#     usedRoute_year = data.query("cate==@c").pivot_table(index='id',
#                                                         columns='year',
#                                                         values='iroute',
#                                                         aggfunc=lambda x: ','.join(x))
#
#     usedRoute_year.to_csv(
#         r'D:\新安\A&H 二周目\Q4\notebook\特徵群組分析\data\profile\usedRoute_{c}.csv'.format(c=c), sep='|')

# %% [markdown]
# y第一個通路 & 最後一個通路

# %%
data.sort_values(['id', 'target', 'date'], inplace=True)
#
#
# data.groupby('id').agg(
#     sex=('sex', lambda x: x.iloc[0]),
#     marr=('marr', lambda x: x.iloc[0]),
#     birthY=('birthY', lambda x: x.iloc[0]),
# )

# %%
data['cate'].unique()

# # %%
# for c in ['CA_PS', 'H_GP', 'H_TV', 'H_PS', 'F_PS']:
#     firstRoute_year = data.query("cate==@c").pivot_table(index=['id'],
#                                                          columns='year',
#                                                          values='iroute',
#                                                          aggfunc=lambda x: x.iloc[0])
#     lastRoute_year = data.query("cate==@c").pivot_table(index='id',
#                                                         columns='year',
#                                                         values='iroute',
#                                                         aggfunc=lambda x: x.iloc[-1])
#
#     firstRoute_year.reset_index().to_csv(
#         r'D:\新安\A&H 二周目\Q4\notebook\特徵群組分析\data\profile\firstRoute_year_{c}.csv'.format(c=c),
#         index=False)
#     lastRoute_year.reset_index().to_csv(
#         r'D:\新安\A&H 二周目\Q4\notebook\特徵群組分析\data\profile\lastRoute_year_{c}.csv'.format(c=c),
#         index=False)


# %%
def iroute_seq(ser, iroute: str):
    init_state = int(iroute in ser.values)
    state = init_state
    opt = 0
    for i in range(1, len(ser)):
        if ser.iloc[i] == ser.iloc[i - 1] == iroute:
            state += 1
        else:
            opt = opt if opt > state else state
            state = init_state
    opt = opt if opt > state else state
    return opt


def iroute_ratio(ser, iroute: str):
    return (ser == iroute).mean()


# %%
def person_info(df, y, cate='CA_PS'):
    basic = df.groupby(['id', 'target']).agg(
        sex=('sex', lambda x: x.iloc[0]),
        marr=('marr', lambda x: x.iloc[0]),
        birthY=('birthY', lambda x: x.iloc[0]),
    )

    ins_bought = df.query('year<=@y')\
        .pivot_table(index=['id', 'target'],
                    columns='cate',
                    values='iply_person',
                    aggfunc=len)
    ins_bought.columns = ['insBought_' + c for c in ins_bought.columns]

    iroute_seqUse = df.query("year<=@y & iroute.isin(@ROUTES) & cate==@cate").groupby(
        ['id', 'target'])['iroute'].agg(
            seqUse_KA=lambda x: iroute_seq(x, 'KA'),
            seqUse_CA=lambda x: iroute_seq(x, 'CA'),
            seqUse_BA=lambda x: iroute_seq(x, 'BA'),
            seqUse_KB=lambda x: iroute_seq(x, 'KB'),
            seqUse_JB=lambda x: iroute_seq(x, 'JB'),
            seqUse_Others=lambda x: iroute_seq(x, 'Others'),
        )

    iroute_prefer = df.query("year<=@y & iroute.isin(@ROUTES) & cate==@cate").groupby(
        ['id', 'target'])['iroute'].agg(
            prefer_KA=lambda x: iroute_ratio(x, 'KA'),
            prefer_CA=lambda x: iroute_ratio(x, 'CA'),
            prefer_BA=lambda x: iroute_ratio(x, 'BA'),
            prefer_KB=lambda x: iroute_ratio(x, 'KB'),
            prefer_JB=lambda x: iroute_ratio(x, 'JB'),
            prefer_Others=lambda x: iroute_ratio(x, 'Others'),
        )

    car_type_cnt = df.query('year==@y & cate==@cate')\
        .pivot_table(index=['id', 'target'],
                    columns='car_type',
                    values='iply_person',
                    aggfunc=len)
    car_type_cnt.columns = ['carType_' + c for c in car_type_cnt.columns]

    car = df.query('year==@y & cate==@cate').groupby(['id', 'target']).agg(car_age=('cur_car_age', 'mean'),
                                                                 car_cnt=('target', 'nunique'))

    contrib = df.query('year==@y').pivot_table(index=['id', 'target'],
                                               columns='cate',
                                               values='plyAmt',
                                               aggfunc=['sum', 'count'])
    contrib.columns = [
        'plyAmt_' + c[1] if c[0] == 'sum' else 'n_order_' + c[1] for c in contrib.columns
    ]

    risk = df.query('year==@y & cate==@cate').groupby(['id', 'target']).agg(
        n_orders=('iply_person', 'nunique'),
        n_clm=('clmAmt', lambda x: (x > 0).sum()),
        clmAmt=('clmAmt', 'sum'),
        clmCnt=('clmCnt', 'sum'),
    ).assign(clmCnt_order=lambda x: x['clmCnt'] / x['n_orders'],
             clmAmt_clmOrder=lambda x: x['clmAmt'] / x['n_clm'])

    ans = pd.concat(
        [basic, ins_bought, iroute_seqUse, iroute_prefer, car_type_cnt, car, contrib, risk], axis=1)
    ans = ans.assign(clmRate=lambda x: x['clmAmt'] / x['plyAmt_' + cate])
    return ans


# %%
for y in range(2017, 2023):

    data['cate'].unique()
    # 車險(強制+任意)
    data_car = data.copy()
    data_car.loc[data_car['cate'].isin(['CA_PS', 'Cf_PS']), 'cate'] = 'CAR'
    data_car['cate'].unique()
    yearly_summary = person_info(data_car, y, cate='CAR')
    yearly_summary.to_csv(
        './data/profile/profile_yearly_summary_CAR_{y}.csv'.format(y=y))
    # 車險(強制)
    yearly_summary = person_info(data.query(" cate!='CA_PS' "), y, cate='Cf_PS')
    yearly_summary.to_csv(
        './data/profile/profile_yearly_summary_CfPS_{y}.csv'.format(y=y))
    # 車險(任意)
    yearly_summary = person_info(data.query(" cate!='Cf_PS' "), y, cate='CA_PS')
    yearly_summary.to_csv(
        './data/profile/profile_yearly_summary_CAPS_{y}.csv'.format(y=y))
