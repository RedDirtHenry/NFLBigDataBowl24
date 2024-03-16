# -----------------------------------------------------------------------------
'''
Data extraction, pre-processing, and calculations.
'''

# -----------------------------------------------------------------------------
# --- Import
import numpy as np
import pandas as pd
import datetime as dt
import os


# Files in Directory
for dirname, _, filenames in os.walk('/Data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# -----------------------------------------------------------------------------
# --- DATA READ
players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/players.csv")
games = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/games.csv")
tackles = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tackles.csv")
plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/plays.csv")
tracking1 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_1.csv")

tracking1["Week"] = 1
tracking2 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_2.csv")
tracking2["Week"] = 2
tracking3 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_3.csv")
tracking3["Week"] = 3
tracking4 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_4.csv")
tracking4["Week"] = 4
tracking5 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_5.csv")
tracking5["Week"] = 5
tracking6 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_6.csv")
tracking6["Week"] = 6
tracking7 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_7.csv")
tracking7["Week"] = 7
tracking8 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_8.csv")
tracking8["Week"] = 8
tracking9 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_9.csv")
tracking9["Week"] = 9



# -----------------------------------------------------------------------------
# --- FOOTBALL CLEANING

# --- Isaiah Simmons position reclassification
mask = players["position"]=="DB"
players.loc[mask,"position"]="ILB"

# --- Homologize Safety to a singular position
mask = (players["position"]=="SS" )| (players["position"]=="FS")
players.loc[mask,"position"] = "S"

# --- Homologize MLB to ILB
mask = (players["position"]=="MLB")
players.loc[mask,"position"] = "ILB"

# --- Homologize NT to DT
mask = (players["position"]=="NT")
players.loc[mask,"position"] = "DT"

# -----------------------------------------------------------------------------
# --- DATA PREPROCESS

# --- merge Plays and Tackles onto Tracking
df = tracking.merge(plays,on=["gameId","playId"],how="left")
df = df.merge(tackles, on=['gameId','playId','nflId'],how='left')

# --- Define Run/Pass Plays
df['RunPass'] = np.where(df['event']=='handoff','Run', 0)
df['RunPass'] = np.where(df['event']=='pass_outcome_caught','Pass', df['RunPass'])
grouping = df[['gameId','playId','RunPass']].groupby(['gameId','playId']).max().reset_index()
df.drop('RunPass', axis=1, inplace=True)
df = df.merge(grouping,on=['gameId','playId'],how='left')

# --- Remove play events that are not needed
invalidPlays = ["lateral","penalty_flag","qb_slide","qb_sack"]
df['Invalid'] = np.where(df['event'].isin(invalidPlays),1,0)
groupInvalid = df[["gameId","playId","Invalid"]].groupby(["gameId","playId"]).max().reset_index()
df.drop("Invalid",axis=1,inplace=True)
df = df.merge(groupInvalid,on=['gameId','playId'],how='left')

df = df[df["Invalid"]==0]

df['RunPass'] = np.where(df['RunPass']=='0', 'Other', df['RunPass'])

# --- Uniform play direction
df = reverse_play_direction(df)

# --- Define Tackle Attempt from event
df['tackle_attempt'] = df.loc[:, ['tackle', 'assist', 'forcedFumble','pff_missedTackle']].sum(axis=1)

# --- Define Defenders
_p = plays.loc[:, ['gameId', 'playId', 'defensiveTeam']].copy()
_p['on_defense'] = 1
df = df.merge(_p, how='left')
df['on_defense'] = df['on_defense'].fillna(0).astype(int)

# --- Ball Location
df['xBallLocation'] = np.where(df['ballCarrierId'] == df['nflId'], df['x'], 0)
df['yBallLocation'] = np.where(df['ballCarrierId'] == df['nflId'], df['y'], 0)
df['xBallLocation'] = df.groupby(['gameId','playId', 'frameId'])['xBallLocation'].transform(max)
df['yBallLocation'] = df.groupby(['gameId','playId', 'frameId'])['yBallLocation'].transform(max)

# --- Ball Distance from each Player
df['ballDistance'] = np.sqrt((df['xBallLocation'] - df['x']) ** 2 + (df['yBallLocation'] - df['y']) ** 2)
