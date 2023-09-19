import numpy as np
import pandas as pd
import plotly.io as pio

import statistics

from sklearn.cluster import KMeans

import plotly.express as ps
import plotly.graph_objs as go

import warnings

warnings.filterwarnings("ignore")

bitcoin = pd.read_csv(
    r"C:\Users\lishe\Desktop\MQF\Term_1\QF627_Programming_Computation_Finance\PyDay\bitcoin_clustering.csv"
)

bitcoin["Date"] = pd.to_datetime(bitcoin["Date"])

bitcoin.set_index(["Date"], inplace=True)

bitcoin_prices = np.array(bitcoin["Adj Close"])

K = 10
kmeans = KMeans(n_clusters=K).fit(bitcoin_prices.reshape(-1, 1))

clusters = kmeans.predict(bitcoin_prices.reshape(-1, 1))

# print(clusters)

pd.options.plotting.backend = "plotly"

pio.templates.default = "none"

cluster_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

fig = bitcoin.plot.scatter(
    x=bitcoin.index,
    y="Adj Close",
    color=[cluster_names[i % len(cluster_names)] for i in clusters],
)

layout = go.Layout(
    showlegend=True,
    plot_bgcolor="white",
    font_color="black",
    font_size=12,
    font_family="Courier",
    xaxis=dict(rangeslider=dict(visible=False)),
)

fig.update_layout(layout)
fig.show()

min_max_values = []

for i in range(K):
    min_max_values.append([np.inf, -np.inf])

for i in range(len(bitcoin_prices)):
    cluster = clusters[i]
    if bitcoin_prices[i] < min_max_values[cluster][0]:
        min_max_values[cluster][0] = bitcoin_prices[i]
    if bitcoin_prices[i] > min_max_values[cluster][1]:
        min_max_values[cluster][1] = bitcoin_prices[i]

output = []

s = sorted(min_max_values, key=lambda x: x[0])

for i, [_min, _max] in enumerate(s):
    if i == 0:
        output.append(_min)

    if i == len(s) - 1:
        output.append(_max)

    else:
        output.append(sum([_max, s[i + 1][0]]) / 2)

fig1 = bitcoin.plot.scatter(
    x=bitcoin.index,
    y="Adj Close",
    color=[cluster_names[i % len(cluster_names)] for i in clusters],
)

for cluster_avg in output[1:-1]:
    fig1.add_hline(y=cluster_avg, line_width=1, line_color="tomato")

fig1.add_trace(
    go.Scatter(
        x=bitcoin.index, y=bitcoin["Adj Close"], line_width=1, line_color="black"
    )
)
layout1 = go.Layout(
    showlegend=True,
    plot_bgcolor="white",
    font_family="Courier",
    font_size=12,
    font_color="black",
    xaxis=dict(rangeslider=dict(visible=False)),
)

fig1.update_layout(layout1)
fig1.show()


### Looking for Optimal K with Elbow Method

values = []

K_range = range(1, 10)

for k in K_range:
    kmeans_n = KMeans(n_clusters=k)
    kmeans_n.fit(bitcoin_prices.reshape(-1, 1))
    values.append(kmeans_n.inertia_)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=list(K_range), y=values, line_width=1, line_color="black"))

layout2 = go.Layout(
    showlegend=True,
    plot_bgcolor="white",
    font_color="black",
    font_size=12,
    font_family="Courier",
    xaxis=dict(rangeslider=dict(visible=False)),
)

fig2.update_layout(layout2)

fig2.show()

# To obtain the size of each cluster

cluster_counts = {}

for c in clusters:
    if c in cluster_counts:
        cluster_counts[c] += 1
    else:
        cluster_counts[c] = 1

# obtain sorted order

cluster_counts = {k: v for k, v in sorted(cluster_counts.items(), key=lambda x: x[0])}

mean = statistics.mean(cluster_counts.values())
sd = statistics.stdev(cluster_counts.values())

print("Mean:", mean)
print("SD:", sd)

kept_clusters = [c for c, v in cluster_counts.items() if v > mean - 0.5 * sd]

fig3 = bitcoin.plot.scatter(
    x=bitcoin.index,
    y="Adj Close",
    color=[cluster_names[i % len(cluster_names)] for i in clusters],
)

layout3 = go.Layout(
    showlegend=True,
    plot_bgcolor="white",
    font_size=12,
    font_family="Courier",
    font_color="black",
    xaxis=dict(rangeslider=dict(visible=False)),
)

for kept in kept_clusters:
    fig3.add_hline(y=output[kept], line_width=1, line_color="tomato")

fig3.add_trace(
    go.Scatter(
        x=bitcoin.index, y=bitcoin["Adj Close"], line_width=1, line_color="black"
    )
)

fig3.update_layout(layout3)
fig3.show()
