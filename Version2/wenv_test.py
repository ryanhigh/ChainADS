import pandas as pd
from wenv4 import CustomEnv
import matplotlib.pyplot as plt
from utils import rns_reward

data  =pd.read_csv('/home/nlsde/RLmodel/Version2/src2/C_matched.csv')

env = CustomEnv()

rl = []
for i in range(len(data)):
    row = data.iloc[i]
    number = row.iloc[5]
    connection = row.iloc[3]
    tps = row.iloc[8]
    delay = row.iloc[11]
    
    se = env.security(number, connection)
    r = env.reward(delay, se, tps)
    rl.append(r)

R = {'reward':rl}
df1 = pd.DataFrame(R)

min_idx = df1['reward'].idxmin()
max_idx = df1['reward'].idxmax()
print(min_idx)
print(max_idx)

sortcounts = df1['reward'].sort_values(ascending=True)
sorl = sortcounts.tolist()
print(sorl)
rns_reward('/home/nlsde/RLmodel/Version2/wenv_test_allr.pkl', 'save', sorl)


"""
奖励值分析, 24 ~ 45
reward   
24.596690    1
24.625371    1
24.977258    1
25.330351    1
25.559765    1
            ..
37.224930    1
37.320338    1
39.227025    1
40.776538    1
45.837355    1
Name: count, Length: 1030, dtype: int64
"""