import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import copy
import time
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',1000)
sys.path.append('/home/xhuang/stockBT_mm/signal_logs')
sys.path.append('/shared/xyang/Data/Reverse_Signal_Data/public_preprocessing')
from label import *

def ensure_path_exsit(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# input price_list volume_list and predict_volume, output how many price count we can eat
# 2.2 -> bid1, bid2 and bid3's 20% volume
def CalMVC(p, v, temp_vol, min_price_move):
    count = 0
    while count < MAX_DEPTH and temp_vol > v[count]:
      temp_vol -= v[count]
      count += 1 # this bid_count bid can be eaten for this volume
    if count >= MAX_DEPTH:
      return abs(round(p[0] - p[MAX_DEPTH-1], 2)/min_price_move)
    else:
      return temp_vol / v[count] +  abs(round(p[0] - p[count], 2) / min_price_move)

def CalPV(df, window_size):  # predict future buy_vol and sell_vol
    df['pvs'] = df['ActiveSell'].rolling(window_size).median().fillna(0)
    df['pvb'] = df['ActiveBuy'].rolling(window_size).median().fillna(0)

def CalOBSC(ob, pre_ob, bid_depth = 1, ask_depth = 1):  # base on this orderbook and last orderbook, calculate the size change among depth interval
    pre_bid1 = sorted(pre_ob['bid'].keys())[-1]
    pre_ask1 = sorted(pre_ob['ask'].keys())[0]
    pre_bid_low_bound = pre_bid1 - bid_depth * min_price_move
    pre_ask_up_bound = pre_ask1 + ask_depth * min_price_move
    pre_bid = copy.deepcopy(pre_ob['bid'])
    pre_ask = copy.deepcopy(pre_ob['ask'])
    bid = copy.deepcopy(ob['bid'])
    ask = copy.deepcopy(ob['ask'])
    bid_pop_list, ask_pop_list, pre_bid_pop_list, pre_ask_pop_list = [], [], [], []
    for k in pre_bid.keys():
        if k < pre_bid_low_bound:
            pre_bid_pop_list.append(k)
    for k in bid.keys():
        if k < pre_bid_low_bound:
            bid_pop_list.append(k)
    for k in pre_ask.keys():
        if k > pre_ask_up_bound:
            pre_ask_pop_list.append(k)
    for k in ask.keys():
        if k > pre_ask_up_bound:
            ask_pop_list.append(k)
    for k in pre_bid_pop_list:
        pre_bid.pop(k)
    for k in pre_ask_pop_list:
        pre_ask.pop(k)
    for k in bid_pop_list:
        bid.pop(k)
    for k in ask_pop_list:
        ask.pop(k)
    pre_bid_array = np.array(sorted(pre_bid.items(), key = lambda x:x[0], reverse=True))
    pre_ask_array = np.array(sorted(pre_ask.items(), key = lambda x:x[0]))
    bid_array = np.array(sorted(bid.items(), key = lambda x:x[0], reverse=True))
    ask_array = np.array(sorted(ask.items(), key = lambda x:x[0]))
    bid_diff = np.sum([i[1] for i in bid_array]) - np.sum([i[1] for i in pre_bid_array])
    ask_diff = np.sum([i[1] for i in ask_array]) - np.sum([i[1] for i in pre_ask_array])
    return bid_diff-ask_diff
#def CalOBSC(df):
  

'''
def show_fail_sample(df, h, side, factor_name):
    if side == 'bid':
        d = df[df[factor_name] >= h]
    else:
        d = df[df[factor_name] <= h]
    fail_d = d[d[side+'_hit'] == False]
    return fail_d.index().tolist()
'''
def show_fail(df, l):
  return df.iloc[l]
  

def pred_signal(df, h, side, factor_name):  # signal accurancy
    if side == 'bid':
        d = df[df[factor_name] >= h]
    else:
        d = df[df[factor_name] <= h]
    return d[side+'_hit'].sum()/len(d)
    #return (d[side+'_ '].sum()/len(d)


def do_signal(df, h, side, factor_name):
    if side == 'bid':
        d = df[df[factor_name] >= h]
    else:
        d = df[df[factor_name] <= h]
    valid_df = df.iloc[signal_freeze(d.index.tolist())]
    return {i:valid_df[valid_df[side+'_status'] == i].index.tolist() for i in [0,1,2]}

#def pred_profit(df, h, side, factor_name):
    #if side == 'bid':
        #d = df[df[factor_name] >= h]
    #else:
        #d = df[df[factor_name] <= h]
    #return len(d[d[side+'_status'] == 1]) / len(d)
#def pred_loss(df, h, side, factor_name):

def signal_times(df, h, side, factor_name):  # num of signal
    if side == 'bid':
        return len(df[df[factor_name] >= h])
    else:
        return len(df[df[factor_name] <= h])

def test_factor(df, factor_list, date, ticker, profit, dir_name = 'png'):  # test factor, plot: 1. acc 2.triggle times with change of threshold, save into directory -dir_name
    fig,ax = plt.subplots(len(factor_list)+1, 4, figsize=(15,8))
    fig.tight_layout(w_pad=5.0)
    ax[0,0].set_title('price move for ap1')
    ax[0,0].plot(df['AP1'], label='ask')
    ax[0,1].set_title('spread move')
    ax[0,1].plot(df['spread'])
    ax[0,2].set_title('tick_volume_move')
    ax[0,2].plot(df['delta_volume'])
    for fl in enumerate(factor_list):
      col = fl[0]+1
      factor_name = fl[1]
      l = np.linspace(df[factor_name].min(), df[factor_name].max(), num=1000)
      up_bound = np.percentile(df[factor_name], 99)
      down_bound = np.percentile(df[factor_name], 1)

      ax[col,0].set_title("factor value move")
      ax[col,0].plot(df[factor_name])
      ax1 = ax[col,0].twinx()
      ax1.plot(df['AP1'], color='red')

      #ax[col,1].set_title("factor distribution")
      #ax[col,1].hist(df[factor_name])

      ax[col,1].plot(l, [len(do_signal(df, i, 'bid', factor_name)[1])/(len(do_signal(df, i, 'bid', factor_name)[1])+len(do_signal(df, i, 'bid', factor_name)[2])) if len(do_signal(df, i, 'bid', factor_name)[1])+len(do_signal(df, i, 'bid', factor_name)[2]) >0 else 1.0 for i in l], label='bid_win')
      minp = l[np.argmin([len(do_signal(df, i, 'bid', factor_name)[1])/(len(do_signal(df, i, 'bid', factor_name)[1])+len(do_signal(df, i, 'bid', factor_name)[2])) if len(do_signal(df, i, 'bid', factor_name)[1])+len(do_signal(df, i, 'bid', factor_name)[2]) >0 else 1.0 for i in l])]
      #axc1 = ax[col,1].twinx()
      #axc1.plot()
      print(do_signal(df, minp, 'bid', factor_name))
      ax[col,1].axhline(bid_hit_ratio, c='red')
      ax[col,1].axhline(ask_hit_ratio, c='black')
      ax[col,1].legend()

      #ax[col,2].plot(l, [len(do_signal(df, i, 'ask', factor_name)[1]) - len(do_signal(df, i, 'ask', factor_name)[2]) for i in l], label='ask_win')
      ax[col,2].plot(l, [len(do_signal(df, i, 'ask', factor_name)[1])/(len(do_signal(df, i, 'ask', factor_name)[1])+len(do_signal(df, i, 'ask', factor_name)[2])) if len(do_signal(df, i, 'ask', factor_name)[1])+len(do_signal(df, i, 'ask', factor_name)[2]) >0 else 1.0 for i in l], label='ask_win')
      #print(do_signal(df, down_bound, 'ask', factor_name))
      ax[col,2].axhline(bid_hit_ratio, c='red')
      ax[col,2].axhline(ask_hit_ratio, c='black')
      ax[col,2].legend()
      ax[col,3].plot(df['AP1'][do_signal(df, minp, 'bid', factor_name)[2][0]:])
      return show_fail(df, do_signal(df, minp, 'bid', factor_name)[2])
      
      ax[col,3].set_title("signal hit times for factor "+factor_name+"@"+str(profit))
      ax[col,3].set_ylim(0,100)
      ax[col,3].plot(l, [signal_times(df, i,'bid', factor_name) for i in l], label='bid')
      ax[col,3].plot(l, [signal_times(df, i,'ask', factor_name) for i in l], label='ask')
      ax[col,3].legend()

    fig.savefig(dir_name + '/' + ticker + "_" + date)

def test_factor1(df, factor_list):
  percentile_pair = [(15,85), (10, 90), (5, 95)]
  percentile_pair = [(5,95)]
  sides = ['bid', 'ask']
  return_df = pd.DataFrame(columns=[s for s in sides], index=[f for f in factor_list])
  for f in factor_list:
    for side in sides:
      for pp in percentile_pair:
        up_bound = np.percentile(df[f], pp[1])
        down_bound = np.percentile(df[f], pp[0])
        result = do_signal(df, up_bound, side, f)
        return_df[side][f] = [len(result[k]) for k in [0, 1, 2]]
  return return_df

'''
def pred(df):  # 
    d=copy.deepcopy(df)
    d = d[d['static_count'] > 2]
    if len(d) ==0: return None
    return (len(d), len(d[d['bid_hit']]) / len(d))
'''

def FilterDF(df):  # filter: remove those bad data
  #df = df[df['AP1'] > 1.0]
  #df = df[df['BP1'] > 1.0]
  df = df[df['AP1'] > df['BP1']]
  for i in range(1,MAX_DEPTH+1):
    df = df[df['AP'+str(i)] > 1.0]
    df = df[df['BP'+str(i)] > 1.0]
  df = df.reset_index()
  return df

def FilterSignal(df):  # filter signal: in these case, should not send signal
  df = df[df['spread'] < 3*min_price_move]
  df = df[df['max_diff'] < 0.11]
  df = df[200:-200]  # head and tail may not be so useful
  df = df.reset_index()
  return df

def InsertFactor(df):  # calculate factor value, append it into original datafrmae
  global min_price_move
  global bid_hit_ratio
  global ask_hit_ratio
  global bid_profit_ratio
  global bid_loss_ratio
  global ask_profit_ratio
  global ask_loss_ratio
  global obsc_size
  df['delta_volume'] = df[' TotalTradeVolume'].diff(1).fillna(0.0)
  df['spread'] = df['AP1'] - df['BP1']
  min_price_move = round(df['spread'][df['spread'] > 0].min(), 2)
  df['max_diff'] = df['AP5'] - df['BP5']
  df['order_book'] = [{'bid':{df['BP'+str(j)][i]:df['BV'+str(j)][i] for j in range(1,6)}, 'ask':{df['AP'+str(j)][i]:df['AV'+str(j)][i] for j in range(1,6)}} for i in range(len(df))]

  '''
  # accurate version based on market_price signal
  df['bid_hit'] = [(np.array(df['ask_future'][i]) <= df['BP1'][i] - min_price_move*profit).sum() > 0 for i in range(len(df))]
  df['ask_hit'] = [(np.array(df['bid_future'][i]) >= df['AP1'][i] + min_price_move*profit).sum() > 0 for i in range(len(df))]
  '''
  AddLabel(df, pred_window_size = 50)
  '''
  df['abuy'] = [df['ActiveBuy'][i - vol_window_size+1:i+1] if i > vol_window_size -1 else df['ActiveBuy'][:i+1] for i in range(len(df))]
  df['asell'] = [df['ActiveSell'][i - vol_window_size+1:i+1] if i > vol_window_size -1 else df['ActiveSell'][:i+1] for i in range(len(df))]
  '''
  CalPV(df, vol_window_size)
  df['bmvc'] = [CalMVC([df['BP'+str(j+1)][i] for j in range(MAX_DEPTH)], [df['BV'+str(j+1)][i] for j in range(MAX_DEPTH)], df['pvs'][i], min_price_move) for i in range(len(df))]
  df['amvc'] = [CalMVC([df['AP'+str(j+1)][i] for j in range(MAX_DEPTH)], [df['AV'+str(j+1)][i] for j in range(MAX_DEPTH)], df['pvb'][i], min_price_move) for i in range(len(df))]
  df['static_count'] = df['bmvc'] - df['amvc']
  #bid_hit_ratio = df['bid_hit'].mean()  # random predict acc
  #ask_hit_ratio = df['ask_hit'].mean()
  bid_profit_ratio = (df['bid_status'] == 1).mean()
  bid_loss_ratio = (df['bid_status'] == 2).mean()
  ask_profit_ratio = (df['ask_status'] == 1).mean()
  ask_loss_ratio = (df['ask_status'] == 2).mean()
  df['obsc'] = [CalOBSC(df['order_book'][i], df['order_book'][i-1]) if i > 0 else 0.0 for i in range(len(df))]
  df['recent_obsc_detail'] = [df['obsc'][i-obsc_size+1:i+1].tolist() if i > obsc_size-1 else df['obsc'][:i+1].tolist() for i in range(len(df))]
  df['recent_obsc'] = [np.average(df['recent_obsc_detail'][i]) for i in range(len(df))]
  df['recent_obsc_sign'] = [np.sign(df['recent_obsc_detail'][i]).sum() for i in range(len(df))]
  df['recent_obsc_sign_ratio'] = [np.sign(df['recent_obsc_detail'][i]).mean() for i in range(len(df))]
  print('bid_hit_ratio is %s'%bid_hit_ratio)
  print('ask_hit_ratio is %s'%ask_hit_ratio)
  print('bid_profit_ratio is %s'%bid_profit_ratio)
  print('bid_loss_ratio is %s'%bid_loss_ratio)
  print('ask_profit_ratio is %s'%ask_profit_ratio)
  print('ask_loss_ratio is %s'%ask_loss_ratio)
  return df

def test(data_file_name, factor_list):  # main function: input file_path, get test results
  start_sec = time.time()
  df = Init(data_file_name)
  df = FilterDF(df)
  df = InsertFactor(df)  ## time_costly 9s
  df = FilterSignal(df)
  print('filter sec is ', time.time()-start_sec)
  start_sec = time.time()
  dir_name = 'png'
  ensure_path_exsit(dir_name)
  #return test_factor(df, factor_list, date, ticker, profit, dir_name)  ## time_costly 19s
  re = test_factor1(df, factor_list)  ## time_costly 19s
  print('test sec is ', time.time()-start_sec)
  return re
  #print(pred(df))

def testall(path, factor_list):
  file_list = os.listdir(path)  # ensure this dir contains data file only
  for f in file_list:
    test(path+'/'+f, factor_list)

# parameters: global variable
file_path = '/home/xhuang/stockBT_mm/signal_logs/20190404_111933_20190315.csv'
#file_path = '/home/xhuang/stockBT_mm/xy_logs'
vol_window_size = 10
pred_window_size = 10
profit = 1
MAX_DEPTH = 5
ticker = '002072'
min_price_move = 0.01
date = ""
ticker=""
bid_hit_ratio = 0.0
ask_hit_ratio = 0.0
obsc_size = 50

def Init(file_name):  # set params and read csv to dataframe
  global vol_window_size
  global pred_window_size
  global profit
  global MAX_DEPTH
  global ticker
  global min_price_move
  global date
  global obsc_size
  print("handling " + file_name)
  date = file_name.split('/')[-1].split('_')[-1].split('.')[0]
  ticker = file_name.split('/')[-1].split('_')[0]
  print("date is ", date)
  print("ticker is ", ticker)
  vol_window_size = 10
  pred_window_size = 20
  profit = 1
  return pd.read_csv(file_name)

def signal_freeze(s, freeze_size = pred_window_size):
  if len(s) == 0:
    return s
  start = s[0]
  l = []
  l.append(start)
  for v in s:
    if v - start > freeze_size:
      l.append(v)
      start = v
  return l

#testall('/home/xhuang/stockBT_mm/xy_logs', ['static_count', 'recent_obsc'])
def main():
  test(file_path, ['static_count'])
  '''
  # parameters: global variable
  file_path = '/home/xhuang/stockBT_mm/signal_logs/20190404_111933_20190315.csv'
  #file_path = '/home/xhuang/stockBT_mm/xy_logs'
  vol_window_size = 10
  pred_window_size = 10
  profit = 1
  MAX_DEPTH = 5
  ticker = '002072'
  min_price_move = 0.01
  date = ""
  ticker=""
  bid_hit_ratio = 0.0
  ask_hit_ratio = 0.0
  obsc_size = 20
  '''

if __name__ == '__main__':
  main()
