# %% import libraries
from datetime import timedelta
from elinimated import general_words
from collections import Counter
from itertools import count
from matplotlib.pyplot import install_repl_displayhook, xticks
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.tools.datetimes import to_datetime
import seaborn as sns
from seaborn.categorical import countplot
sns.set(style='darkgrid')

# %% set up some regexs and date format to correctly parse chat file
regex_iphone = r".?\[(\d{1,2}.\d{1,2}.\d{4}\s\d{1,2}:\d{1,2}:\d{1,2})\]\s([\w\s]+):\s(.+?)(?=.?.?\[\d{1,2}.\d{1,2}.\d{4}\s\d{1,2}:\d{1,2}:\d{1,2}\])$"
date_format_iphone = '%d.%m.%Y %H:%M:%S'

regex_android = r"(\d{1,2}.\d{1,2}.\d{4}\s\d{1,2}:\d{1,2})\s-\s([\w\s]+):\s(.+?)(?=.?.?\d{1,2}.\d{1,2}.\d{4}\s\d{1,2}:\d{1,2})$"
date_format_android = '%d.%m.%Y %H:%M'

# %% read the chat file
file_name = 'data/chat.txt'

# read the file
with open(file_name, 'r') as file:
    text = file.read()

# parse the data as if exported from an iphone
lines = re.findall(regex_iphone, text, re.DOTALL |
                   re.IGNORECASE | re.MULTILINE)

# if former approach doesn't work, assume it is android.
if not lines:
    lines = re.findall(regex_android, text, re.DOTALL |
                       re.IGNORECASE | re.MULTILINE)

# %% convert to dataframe and set index
# also date column needs to be pandas datatime,
# so we are converting to it
columns = ['date', 'name', 'message']
df = pd.DataFrame(lines, columns=columns)
df.set_index('name', drop=True, inplace=True)
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

# %% we exported chat data as 'no media' so, we need to remove placeholders from data
placeholders = [
    'görüntü dahil edilmedi',
    'Çıkartma dahil edilmedi',
    'ses dahil edilmedi',
    '‎GIF dahil edilmedi',
    '‎video dahil edilmedi',
    '‎belge dahil edilmedi',
    '<Medya dahil edilmedi>',
    '‎Mesajlar ve aramalar uçtan uca şifrelidir']

df = df[df["message"].str.contains('|'.join(placeholders)) == False]
df[df['message'].str.contains('dahil edilmedi')].any()

# %% print total dates
print("Days Total: ", (df['date'][-1] - df['date'][0]).days)

# %% print message counts
people = df.index.unique()
for person in people:
    print(f"{person} has sent {len(df.loc[person])} messages")
print(f"Total {len(df)} messages has been sent")

# %% resample by month and calculate message count per month
df_date = df.copy().reset_index().set_index('date')

df_sent = df_date.groupby('name').resample('M').count()
df_sent.drop('name', inplace=True, axis=1)

df_sent = df_sent.unstack(0).droplevel(0, axis=1)

# %% utilty method to labelize plot's x axis
def year_and_month(x):
    return str(x.year) + ', ' + str(x.month_name())

# %% plot message count per month
# __bar_plot__
ax = df_sent.plot(kind='bar')
ax.set(title='Message Count per Month', xticklabels=df_sent.index.map(year_and_month),
       xlabel='Month', ylabel='Message Count')

# %% plot message count per month
# __line_plot__
melted = df_sent.melt()
fig, ax = plt.subplots()
for person in people:
    sns.lineplot(data=melted[melted['name'] == person],
                 x=df_sent.index.map(year_and_month), y='value', ax=ax)

ax.set(title='Message Count per Month', xlabel='Month', ylabel='Message Count')

for label in ax.get_xticklabels():
    label.set_rotation(90)
    label.set_ha('right')

ax.legend(labels=people)

# %% calculate messages per hour
df_hour = df.copy()
df_hour['date'] = df_hour['date'].apply(lambda x: x.hour)
df_hour = df_hour.groupby(['date', 'name'])['message'].count().unstack(0).T

# %% plot messages per hour
# __bar_plot__
ax = df_hour.plot(kind='bar')
ax.set(title='Total Message Count per Hour',
       xlabel='Hour', ylabel='Message Count')

# %% plot messages per hour
# __line_plot__
ax = df_hour.plot(kind='line')
ax.set(title='Total Message Count per Hour', xticks=range(
    0, len(df_hour), 3), xlabel='Hour', ylabel='Messages Count')

# %% calculate messages per day name
df_day = df.copy()
df_day['date'] = df_day['date'].apply(lambda x: x.day_name())
df_day = df_day.groupby(['date', 'name'])['message'].count().unstack(0).T

# %% plot messages per day names
ax = df_day.plot(kind='bar')
ax.set(title='Total Message Count per Day Name',
       xlabel='Day', ylabel='Messages Sent')

# %% calculate word count
df_count = df.copy()
df_count['date'] = df_count['date'].apply(lambda x: x.date())
wpm = []
for person in people:
    word_count = df_count.loc[person].groupby('date').apply(
        lambda x: np.sum(x['message'].str.split().str.len()))
    total_words = np.sum(word_count)
    wpm.append(total_words//38)
    print(f"{person} has sent {total_words} total words")

# %%
print("On mobile devices, the average typing speed is 38 word per minute. Based on this,")
for idx, person in enumerate(people):
    elapsed = "{:0>8}".format(str(timedelta(seconds=wpm[idx].item()*60)))
    print(f"{person} has spent nearly {elapsed}")

# %% Print first and last day
first_mdate = df['date'].iloc[0].strftime('%b %d, %Y at %H:%S:%M')
last_mdate = df['date'].iloc[-1].strftime('%b %d, %Y at %H:%S:%M')
print(f"Messages are,\nFrom {first_mdate} to {last_mdate}")

# %% calculate most active day
active = df_count.groupby('date').apply(lambda x: len(x))
active_count = active.max()
active_date = active.idxmax().strftime('%b %d, %Y')
print(
    f"Most active day is {active_date}. That day, {active_count} messages has been sent")

# %% calculate average word length in messages per month
avg_word_month = df_date.groupby('name').resample(
    'M').apply(lambda x: np.mean(x.str.split().str.len()))
avg_word_month = avg_word_month.drop('name', axis=1)
avg_word_month = avg_word_month.unstack(0).droplevel(0, axis=1)

# %% plot average word length in messages per month
ax = avg_word_month.plot()
ax.set(title='Average Word Count in a Message per Month', xticklabels=avg_word_month.index.map(year_and_month),
       xlabel='Month', ylabel='Word Count')

# %% calculate average messages per day and month
df_avg_day = df_count.groupby('date').count()
df_avg_day.index = df_avg_day.index.map(lambda x: pd.to_datetime(x))
df_avg_month = df_avg_day.resample('M').apply(lambda x: np.mean(x))

# %% plot average messages per month
ax = df_avg_month.plot(kind='line')
ax.set(title='Message Count per Month', xlabel='Month', ylabel='Message Count')

# %% plot average messages per day
ax = df_avg_day.abs().plot.area(grid=1, linewidth=0.5, rot=90)
ax.set(title='Message Count per Day', xlabel='Month', ylabel='Message Count')

# %% finds most used words
df_freq = df.drop('date', axis=1)
most_used = {}
for person in people:
    text = ' '.join(df_freq.loc[person, 'message'].str.lower().values)
    counter = [x for x in Counter(text.split()).most_common(
        250) if x[0] not in general_words]
    most_used.update({person: counter})

# %% print most used words
df_most_used = pd.DataFrame.from_dict(
    most_used, orient='index').T.replace('None', np.nan).dropna()

# below method used to print full dataframe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_most_used)
