from YahooDataset import YahooDataset
import matplotlib.pyplot as plt, mpld3
from collections import defaultdict

# font = {'weight' : 'bold'}
# plt.rc('font', **font)
yd = YahooDataset()
# trainSet = yd.loadFullSet()
# ratings = defaultdict(int)
# for uid, iid, rating in trainSet.all_ratings():
#     ratings[rating] += 1
#
# ratings = dict(sorted(ratings.items()))
#
# width = 0.8
# labels = list(range(1, len(ratings.keys())+1))
# values = list(ratings.values())
#
# _, ax = plt.subplots()
# rects1 = ax.bar(labels, ratings.values(), width)
# ax.set_ylabel('Učestalost', fontsize=20)
# ax.set_xlabel('Rejting', fontsize=20)
# ax.set_title('Raspodela rejtinga u apsolutnim brojevima', fontsize=25)
# ax.set_xticks(labels)
# plt.tick_params(axis='both', which='minor', labelsize=30)
#
# figure = plt.gcf()
#
# for i, bar in enumerate(rects1.get_children()):
#     tooltip = mpld3.plugins.LineLabelTooltip(bar, label=values[i])
#     mpld3.plugins.connect(figure, tooltip)

# x = values[-5:]
# x.append(sum(values[:8]))
# explode = [0 for _ in range(len(x)-1)]
# explode.append(0.2)
# y = labels[-5:]
# y.append("Ostalo")
# plt.pie(x,labels=y,autopct='%1.1f%%', explode = explode, textprops={'fontsize': 20})
# plt.title("Raspodela rejtinga u relativnim brojevima", fontsize=25)
#
# demographicData = yd.loadDemographicsData()
# demographicStats = demographicData.groupby('gender').count().reset_index().rename(columns={'userId':'count'})
#
# ax = demographicStats.plot.bar(x='gender', y='count', rot=0, legend=False)
# ax.set_title('Raspodela korisnika prema polu', fontsize=25)
# ax.set_xlabel("Pol", fontsize=20)
# ax.set_ylabel("Broj korisnika", fontsize=20)
#
# figure = plt.gcf()
# for i, bar in enumerate(ax.patches):
#     tooltip = mpld3.plugins.LineLabelTooltip(bar, label=demographicStats.iloc[i, 1])
#     mpld3.plugins.connect(figure, tooltip)
#
# otherStats = yd.loadYahooPandasFullDataFrame().merge(demographicData, left_on="userId", right_on="userId").groupby("gender", as_index=False).agg({"rating":"count"})
# ax = otherStats.plot.bar(x='gender', y='rating', rot=0, legend=False)
# ax.set_title("Raspodela rejtinga prema polu", fontsize=25)
# ax.set_xlabel("Pol", fontsize=20)
# ax.set_ylabel("Broj rejtinga", fontsize=20)
#
# figure = plt.gcf()
#
# for i, bar in enumerate(ax.patches):
#     tooltip = mpld3.plugins.LineLabelTooltip(bar, label=otherStats.iloc[i, 1])
#     mpld3.plugins.connect(figure, tooltip)


# mpld3.save_html(figure, "figura.html", d3_url="static/js/d3.v3.min.js", mpld3_url="static/js/mpld3.v0.3.js")

# data = self.fullDataFrame
#
# data = data[data["movieId"] == 0]
# data = data.loc[:, ["userId", "rating"]].groupby('rating').count().reset_index().rename(columns={'userId': 'count'})

# ax = data.plot.bar(x='rating', y='count', rot=0, legend=False)
# ax.set_title('Raspodela rejtinga', fontsize=20)
# ax.set_xlabel("Rejting", fontsize=15)
# ax.set_ylabel("Ucestalost", fontsize=15)


trainSet = yd.loadFullSet()

movieId = trainSet.to_inner_iid("1800340949")
ratings = defaultdict(int)
for _, rating in trainSet.ir[movieId]:
    ratings[int(rating)] += 1

ratings = dict(sorted(ratings.items()))

_, ax = plt.subplots()

ax.bar(range(len(ratings)), ratings.values(), align='center')
plt.xticks(range(len(ratings)), list(ratings.keys()))
plt.title("Raspodela rejtinga")
plt.xlabel("Rejting")
plt.ylabel("Učestalost")

plt.show()

# ax = plt.gca()
figure = plt.gcf()
for i, bar in enumerate(ax.patches):
    tooltip = mpld3.plugins.LineLabelTooltip(bar, label=list(ratings.values())[i])
    mpld3.plugins.connect(figure, tooltip)
