"""
=======================================
Visualizing the stock market structure
=======================================

This example employs several unsupervised learning techniques to extract
the stock market structure from variations in historical quotes.

The quantity that we use is the daily variation in quote price: quotes
that are linked tend to cofluctuate during a day.

.. _stock_market:

Learning a graph structure
--------------------------

We use sparse inverse covariance estimation to find which quotes are
correlated conditionally on the others. Specifically, sparse inverse
covariance gives us a graph, that is a list of connection. For each
symbol, the symbols that it is connected too are those useful to explain
its fluctuations.

Clustering
----------

We use clustering to group together quotes that behave similarly. Here,
amongst the :ref:`various clustering techniques <clustering>` available
in the scikit-learn, we use :ref:`affinity_propagation` as it does
not enforce equal-size clusters, and it can choose automatically the
number of clusters from the data.

Note that this gives us a different indication than the graph, as the
graph reflects conditional relations between variables, while the
clustering reflects marginal properties: variables clustered together can
be considered as having a similar impact at the level of the full stock
market.

Embedding in 2D space
---------------------

For visualization purposes, we need to lay out the different symbols on a
2D canvas. For this we use :ref:`manifold` techniques to retrieve 2D
embedding.


Visualization
-------------

The output of the 3 models are combined in a 2D graph where nodes
represents the stocks and edges the:

- cluster labels are used to define the color of the nodes
- the sparse covariance model is used to display the strength of the edges
- the 2D embedding is used to position the nodes in the plan

This example has a fair amount of visualization-related code, as
visualization is crucial here to display the graph. One of the challenge
is to position the labels minimizing overlap. For this we use an
heuristic based on the direction of the nearest neighbor along each
axis.
"""

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

import sys
import os
# import requests

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

from sqlconn import engine


# from sqlrw import readTTMPE

# print(__doc__)


def downChengfen():
    """
    下载指数成份股，月度数据
    :return:
    """
    pro = ts.pro_api()
    ID = '000016.SH'
    # df = pro.index_weight(index_code=ID, trade_date='20190105')
    df = pro.index_weight(index_code=ID, start_date='20190601',
                          end_date='20190630')
    df['stockid'] = df.con_code.str[:6]
    df = df[['stockid']]
    df.to_csv('index_chengfen.csv')
    print(df.head())


def readStockListFromSQL():
    result = engine.execute('select stockid, name from stocklist '
                            'where timetomarket>0')
    stocks = [i for i in result]
    # for id, name in result:
    #     stocks[id] = name
    return stocks


def readStockID():
    df1 = pd.read_csv('index_chengfen.csv')


def myplot():
    df = pd.read_csv('index_chengfen.csv')
    # print(df)
    symbols = df.stockid.to_list()
    names = df.name.to_list()
    # print(symbols)
    # print(names)


def readPE():
    """
    返回每支股票每日pe涨跌，每股票占一行
    :return:
    """
    res = None
    ids = ['600000', '600016', '600019', '600028']
    lens = len(ids)
    for i in range(lens):
        stockID = ids[i]
        print(f'process {stockID}, {i + 1}/{lens}')
        df = _readPE(stockID)
        if res is None:
            res = df.copy()
        else:
            res = res.join(df, how='outer')
    # res.fill(0)
    res = res[1:]
    res = res.fillna(0)
    res = res[ids]
    print(res.head())
    print(res[res.isnull().values])
    print(res.shape)
    cols = res.columns
    print(type(cols), cols)
    print(res.values.T.tolist())
    return res.values.T.tolist()


def _readPE(stockID='600000', startDate='20180101', endDate='20181231'):
    """
    返回某支股票每日pe涨跌
    :return:
    """
    sql = (f'select date, ttmpe from klinestock where stockid={stockID} '
           f'and date>={startDate} and date<={endDate}')
    res = engine.execute(sql)
    pes = [i for i in res]
    # print(pes)
    df = pd.DataFrame(pes, columns=['date', 'pe'])
    # print(df.head())
    peB = np.array(df.pe[1:])
    peA = np.array(df.pe[:-1])
    peC = peB - peA
    peC = np.insert(peC, 0, 0)

    # print(peA.size, peA)
    # print(peB.size, peB)
    # print(peC.size, peC)

    # 每日pe较上一日的涨跌值
    df[stockID] = peC
    df = df[['date', stockID]]
    df = df.set_index('date')
    # print(df.head())
    return df


def plotStockMarket():
    # #############################################################################
    # Retrieve the data from Internet

    # The data is from 2003 - 2008. This is reasonably calm: (not too long ago
    # so that we get high-tech firms, and before the 2008 crash). This kind of
    # historical data can be obtained for from APIs like the quandl.com and
    # alphavantage.co ones.

    symbol_dict = {
        'TOT': 'Total',
        'XOM': 'Exxon',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'VLO': 'Valero Energy',
        'MSFT': 'Microsoft',
        'IBM': 'IBM',
        'TWX': 'Time Warner',
        'CMCSA': 'Comcast',
        'CVC': 'Cablevision',
        'YHOO': 'Yahoo',
        'DELL': 'Dell',
        'HPQ': 'HP',
        'AMZN': 'Amazon',
        'TM': 'Toyota',
        'CAJ': 'Canon',
        'SNE': 'Sony',
        'F': 'Ford',
        'HMC': 'Honda',
        'NAV': 'Navistar',
        'NOC': 'Northrop Grumman',
        'BA': 'Boeing',
        'KO': 'Coca Cola',
        'MMM': '3M',
        'MCD': 'McDonald\'s',
        'PEP': 'Pepsi',
        'K': 'Kellogg',
        'UN': 'Unilever',
        'MAR': 'Marriott',
        'PG': 'Procter Gamble',
        'CL': 'Colgate-Palmolive',
        'GE': 'General Electrics',
        'WFC': 'Wells Fargo',
        'JPM': 'JPMorgan Chase',
        'AIG': 'AIG',
        'AXP': 'American express',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'AAPL': 'Apple',
        'SAP': 'SAP',
        'CSCO': 'Cisco',
        'TXN': 'Texas Instruments',
        'XRX': 'Xerox',
        'WMT': 'Wal-Mart',
        'HD': 'Home Depot',
        'GSK': 'GlaxoSmithKline',
        'PFE': 'Pfizer',
        'SNY': 'Sanofi-Aventis',
        'NVS': 'Novartis',
        'KMB': 'Kimberly-Clark',
        'R': 'Ryder',
        'GD': 'General Dynamics',
        'RTN': 'Raytheon',
        'CVS': 'CVS',
        'CAT': 'Caterpillar',
        'DD': 'DuPont de Nemours'}

    symbols, names = np.array(sorted(symbol_dict.items())).T

    quotes = []

    for symbol in symbols:
        print('Fetching quote history for %r' % symbol, file=sys.stderr)
        # url = ('https://raw.githubusercontent.com/scikit-learn/examples-data/'
        #        'master/financial-data/{}.csv')
        # df = pd.read_csv(url.format(symbol))
        # df.to_csv(filename)
        # quotes.append(pd.read_csv(url.format(symbol)))

        filename = os.path.join('data', f'{symbol}.csv')
        df = pd.read_csv(filename)
        quotes.append(df)
        # print(df)

    # sys.exit(0)

    close_prices = np.vstack([q['close'] for q in quotes])
    open_prices = np.vstack([q['open'] for q in quotes])
    # print('close_prices:')
    # print(close_prices)

    # The daily variations of the quotes are what carry most information
    variation = close_prices - open_prices
    print(variation)

    # #############################################################################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphicalLassoCV(cv=5)

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    # #############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    # #########################################################################
    # Find a low-dimension embedding for visualization: find the best position
    # of the nodes (the stocks) on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T
    # embedding = node_position_model.fit_transform(X.T)

    # #############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    # noinspection PyTypeChecker
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.nipy_spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(
                               label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(), )
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()


if __name__ == '__main__':
    pass
    # downChengfen()
    # myplot()
    readPE()
    # plotStockMarket()
