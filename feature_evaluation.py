import os
import pandas as pd
from importlib import reload
from pyscorecard.context import DataContext
from  pyscorecard.plot import plot_bin_trend as pbt
from pyscorecard.tools import split_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tool_20p as tl
import SwapAuto_V2 as sw
import cpca
root_dir = r"C:\Users\zhongzifeng\PycharmProjects\untitled\data\iflowmodel"
raw=pd.read_csv(r"C:\Users\zhongzifeng\PycharmProjects\untitled\data\iflowmodel\zzf_1226.csv",dtype={'cid':str})
raw=raw.sort_values(by='update_time')
raw=raw.drop_duplicates(subset='cid',keep='last')
raw['cid']=raw['cid'].map(lambda x:int(x.split('.')[0]))
ym=pd.read_csv(os.path.join(root_dir,'秒啦_code0.csv'),index_col=0)
ym.columns=['id_type', 'loan_date', 'cid', 'credit_score1']
raw=pd.merge(raw,ym[['cid', 'credit_score1']],on='cid',how='left')
raw['dz_credit_score']=raw[['dz_credit_score','credit_score1']].mean(axis=1)
del raw['credit_score1']
# raw['audit_time']=raw['audit_time'].map(lambda x:x[:10])
# raw=raw[raw.audit_time>='2019-09-11']
# raw=raw.sort_values(by='audit_time').reset_index(drop=True)
# raw['y']=raw['late_days'].map(lambda  x:1 if x>7 else 0)
raw['add_time']=raw['add_time'].map(lambda x:x[:10])
# raw=raw[raw.add_time>='2019-09-11']
raw=raw.sort_values(by='add_time').reset_index(drop=True)
raw['bal_amt'] = raw['loan_bal']/raw['loan_amt']
raw['bal_amt']=raw['bal_amt'].replace(np.inf,np.nan)
raw['amt_account'] = raw['loan_amt']/raw['loan_account']
raw['amt_account']=raw['amt_account'].replace(np.inf,np.nan)
raw.ali_score = raw.ali_score.replace(0, np.nan)


var_use1=['tj_score',  'dz_credit_score',
        'ali_score',  'query_total_org',
       'industry',  'age', 'amt_account','phone_province',
       'verif_count_m1', 'repay_remind_sum_ratio_m3m6',
       'repay_remind_cnt_ratio_m1m12', 'verif_sum_ratio_m1m12', 'debt',
       'max_overdue_repay_amount_level_m6', 'apply_request_sum_ratio_m1m12',
       'nre_d90_maxloanamount', 'bal_amt', 'max_loan_offer_amount_level_m3',
         ]#去掉省份20
var_use=['tj_score',  'dz_credit_score',
        'ali_score','age',
         # 'age&account','age&amt',
       'industry','phone_province',
       'verif_count_m1','bal_amt', 'max_loan_offer_amount_level_m3',
         'degree','id_validity_years','nre_smy_loancount',
         'nre_smy_overduecount','re_d360_lendingamount']#单调性整合
var_use=['als_m12_cell_bank_max_inteday', 'als_m6_cell_max_inteday',
       'als_fst_id_bank_inteday', 'als_m12_id_tot_mons',
       'als_m12_cell_nbank_avg_monnum', 'als_m12_cell_avg_monnum',
       'als_lst_id_nbank_inteday', 'als_m12_id_nbank_max_inteday',
       'als_m12_id_nbank_night_allnum', 'als_m6_id_max_inteday',
       'als_m12_cell_max_inteday', 'als_m12_id_nbank_allnum',
       'als_fst_cell_bank_inteday', 'als_m3_id_nbank_oth_allnum',
       'als_m12_cell_nbank_night_orgnum', 'als_m3_id_nbank_orgnum',
       'als_m12_cell_caoff_allnum', 'als_m12_cell_nbank_cf_allnum',
       'als_m12_cell_nbank_nsloan_orgnum', 'als_m6_id_nbank_sloan_allnum',
       'als_m3_cell_nbank_else_allnum', 'als_m12_id_rel_orgnum']

remark={'age':'年龄','ali_score':'阿里分'}
type_dc=tl.Widget.bin_type_guess(raw[var_use])
data_dir=r"C:\Users\zhongzifeng\PycharmProjects\untitled\data\iflowmodel"
dc = DataContext(y="y",
                 train=raw.loc[raw.inflow_flag==1,var_use+['audit_time','y']],
                 test=raw.loc[raw.source_channel == 21,var_use+['audit_time','y']],
                 oot=raw.loc[(raw.source_channel == 3)&(raw.inflow_flag!=1),var_use+['audit_time','y']],
                 dtypes=type_dc,
                 data_dir=data_dir,
                 data_type = {'train': '信息流', 'test': '榕树', 'oot': 'IOS'},time_column='audit_time')
dc = DataContext(y="y",
                 train=raw.loc[(raw.inflow_flag==1)&(raw.audit_time <= '2019-11-22'),var_use+['y']],
                 test=raw.loc[(raw.inflow_flag==1)&(raw.audit_time >= '2019-11-22'),var_use+['y']],
                 oot=raw.loc[(raw.inflow_flag==1)&(raw.audit_time >= '2019-11-22'),var_use+['y']],
                 dtypes=type_dc,
                 data_dir=data_dir)
dc.bin_version = "Info_flow_v3"
# dc.binning_init(MDP=True, end_num=5, mono=False, method=3,bin_num=10)
cr = pd.read_excel(os.path.join(root_dir, "0.5ar.xlsx"),'cr')
# cr=cr[cr.Feature.isin(var_use)]
dc.binning_from_change(cr,remark=None)
df=raw[(raw.inflow_flag!=1)&(raw.source_channel==3)]
df.audit_time=df.audit_time.map(lambda x:int(x.replace('-','')))
df.audit_time=pd.qcut(df.audit_time,12)
pt=pbt.BinTrendPlot(raw,var_use1,'y','audit_time',cr)
pt.plot_bin(os.path.join(root_dir,'bin_trend_ios.html'))

# ********************************************************************************************/
def my_split_left(x):
    if pd.isnull(x) or x == 'missing':
        return np.nan
    else:
        try:
            a = x.split(',')[0].replace('(', '').replace('[', '')
            if a == '-inf':
                return -np.inf
            else:
                return float(a)
        except:
            return x.replace('(', '').replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(',')


def my_split_right(x):
    if pd.isnull(x) or x == 'missing':
        return np.nan
    else:
        try:
            a = x.split(',')[1].replace(']', '').replace(')', '')
            if a == 'inf':
                return np.inf
            else:
                return float(x.split(',')[1].replace(']', '').replace(')', ''))
        except:
            return x.replace('(', '').replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(',')


def my_interval(x):
    if isinstance(x['lower'], list) == False and pd.isnull(x['lower']):
        return np.nan
    elif x['lower'] == x['upper']:
        return x['upper']
    else:
        return pd.Interval(x['lower'], x['upper'], closed='right')


# 单个SERIES
def woe_replace_dis(x, bin_map, left=False):
    y = pd.Series(np.nan, index=x.index)
    extra = np.nan
    for i in range(0, len(bin_map)):
        if isinstance(bin_map['Interval'].iloc[i], list):
            y[(x.isin(bin_map['Interval'].iloc[i]))] = bin_map['Score'].iloc[i]
        elif pd.isnull(bin_map['Interval'].iloc[i]) or bin_map['Interval'].iloc[i] == ['NaN']:
            extra = bin_map['Score'].iloc[i]
        else:
            if left:
                y[(x >= bin_map['Interval'].iloc[i].left) & (x < bin_map['Interval'].iloc[i].right)] = \
                bin_map['Score'].iloc[i]
            else:
                y[(x > bin_map['Interval'].iloc[i].left) & (x <= bin_map['Interval'].iloc[i].right)] = \
                bin_map['Score'].iloc[i]
    y = y.fillna(extra)
    return y


# dataframe 转换
def woe_replace_df(df, var, score, left=False):
    tran_df = {}
    for x in var:
        print(x)
        bin_map = score[score['Feature'] == x].reset_index(drop=True)
        try:
            tran_df[x] = woe_replace_dis(df[x], bin_map, left=left)
        except:
            tran_df[x] = woe_replace_dis(df[x].astype(float), bin_map, left=left)
        tran_df[x].name = x
    df_tmp = pd.concat([tran_df[x] for x in tran_df], axis=1)
    return df_tmp


# ********************************************************************************************/
''' mljk '''
# ------------------------------------------------------------------------------------------/
mljy_dict = pd.read_excel("/Users/bacallzhong/PycharmProjects/data/sc162/br_var.xlsx",'bin',index_col=0)
mljy_dict['lower'] = mljy_dict['Interval'].map(lambda x: my_split_left(x))
mljy_dict['upper'] = mljy_dict['Interval'].map(lambda x: my_split_right(x))
mljy_dict['Interval'] = mljy_dict.apply(my_interval, axis=1)

# ------------------------------------------------------------------------------------------
var_use=['als_m12_cell_bank_max_inteday', 'als_m6_cell_max_inteday',
       'als_fst_id_bank_inteday', 'als_m12_id_tot_mons',
       'als_m12_cell_nbank_avg_monnum', 'als_m12_cell_avg_monnum',
       'als_lst_id_nbank_inteday', 'als_m12_id_nbank_max_inteday',
       'als_m12_id_nbank_night_allnum', 'als_m6_id_max_inteday',
       'als_m12_cell_max_inteday', 'als_m12_id_nbank_allnum',
       'als_fst_cell_bank_inteday', 'als_m3_id_nbank_oth_allnum',
       'als_m12_cell_nbank_night_orgnum', 'als_m3_id_nbank_orgnum',
       'als_m12_cell_caoff_allnum', 'als_m12_cell_nbank_cf_allnum',
       'als_m12_cell_nbank_nsloan_orgnum', 'als_m6_id_nbank_sloan_allnum',
       'als_m3_cell_nbank_else_allnum', 'als_m12_id_rel_orgnum']
df=pd.read_csv('/Users/bacallzhong/PycharmProjects/data/sc162/br进件.tsv',sep='\t')
df.apply_time=df.apply_time.map(lambda x:int(x.replace('-','')))
df.apply_time=pd.qcut(df.apply_time,16)
var = mljy_dict['Feature'].unique()
df_tmp = woe_replace_df(raw,var_use,mljy_dict)
df_tmp['apply_time']=df['apply_time']
from pyscorecard.plot.plot_long_trend import LongTrend
lt = LongTrend(df_tmp,var_use,'apply_time',os.path.join(root_dir,'long_trend_ios.html'))
lt.plot()



#-------------------------------------------
score = pd.read_csv(os.path.join(root_dir,'0911_1024_score.csv'),index_col=0)
score=score[score.apply_time>=20191011]
save_dir=r"C:\Users\zhongzifeng\PycharmProjects\model\newmodel1\首贷模型报告12_5_v3\swap_oot"
sw.Swap(score,'y','score_141','score_157','','',4,4,'157_swap_141.xls',save_dir,10,10)
sw.Swap(score,'y','score_138','score_157','','',4,4,'157_swap_138.xls',save_dir,10,10)
sw.Swap(score,'y','score_138','final_score','','',4,4,'score_swap_138.xls',save_dir,10,10)
sw.Swap(score,'y','final_score','score_157','','',4,4,'score_swap_157.xls',save_dir,10,10)
# sw.Swap(score,'y','score_141','score_138','','','score_141','score_138',4,4,'138_swap_141.xls',save_dir,10,10)
score['flag'] = score['channel']
score_1 = score.loc[score.channel.isin([3,21])]
sw.Swap(score_1,'y','score_141','score_157','','',4,4,'157_swap_141_bychannel.xls',save_dir,10,10)
sw.Swap(score_1,'y','score_138','score_157','','',4,4,'157_swap_138_bychannel.xls',save_dir,10,10)
sw.Swap(score_1,'y','score_138','final_score','','',4,4,'score_swap_138_bychannel.xls',save_dir,10,10)
sw.Swap(score_1,'y','final_score','score_157','','',4,4,'score_swap_157_bychannel.xls',save_dir,10,10)
# sw.Swap(score_1,'y','score_141','score_138','','','score_141','score_138',4,4,'138_swap_141_bychannel.xls',save_dir,10,10)
