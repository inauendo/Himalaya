import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def EDA_peaks(exp, plotpath = None):
    '''Performs EDA comparing ascension attempts and successes per peak in the region.'''
    #Create a view comparing successful and successful expeditions per peak
    peakview = exp.groupby('peakid').success.agg(['sum', 'count'])
    peakview = peakview.rename(columns={'sum': 'successes', 'count': 'total'})
    peakview['ratio'] = peakview.successes / peakview.total

    #Let us focus on the 30 peaks which have the highest amount of ascension attempts
    cut = peakview.sort_values('total', ascending=False).iloc[:30]

    plt.clf()
    rects = plt.bar(x = cut.index, height = cut.total, color='r', alpha=1, label='Failed attempts')
    plt.bar(x = cut.index, height = cut.successes, color='k', alpha=1, label='Successful attempts')
    for i in range(len(rects)):
        plt.text(rects[i].get_x() + rects[i].get_width()/2, rects[i].get_height(), '{:.2f}'.format(cut.iloc[i].ratio), va='bottom', ha='center')
    ticks, _ = plt.xticks()
    plt.xticks(ticks, cut.index, rotation='vertical')
    plt.legend()
    plt.title('Ascension attempts and success rates for the 30 most often attempted Himalayan peaks')
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/ascensions_per_peak.png', dpi=640)

    #The Figure shows that the distribution is heavily skewed. Let's print some statistics:
    print("Median ascencion attempts for every peak: {}".format(peakview.total.median()))
    print("Everest accounts for {:.2%} of all ascension attempts".format(cut.iloc[0].total/peakview.total.sum()))
    print("The top 10 peaks (out of 398) account for {:.2%} of all ascension attempts".format(cut.total.iloc[:10].sum()/peakview.total.sum()))

    attemptlimit = 5
    plt.clf()
    plt.hist(peakview.loc[peakview.total > attemptlimit].ratio, bins = 14)
    upper, lower = plt.ylim()
    plt.plot(np.ones(100)*peakview.ratio.mean(), np.linspace(0,30,100), 'r--', label='mean')
    plt.ylim(upper, lower)
    plt.legend()
    plt.title('Success ratio distribution of the {0} peaks featuring > {1} attempts'.format(peakview.loc[peakview.total > 10].ratio.count(), attemptlimit))
    plt.xlabel('Success ratio')
    plt.xlim(0,1)
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/successratio_distribution.png', dpi=640)

def EDA_year(exp, plotpath = None):
    '''perform EDA for ascensions per year.'''
    #Create a view comparing successful and successful expeditions per year
    yearview = exp.groupby('year').success.agg(['sum', 'count'])
    yearview = yearview.rename(columns={'sum': 'successes', 'count': 'total'})
    yearview['ratio'] = yearview.successes / yearview.total

    #fill in missing values
    new_index = np.arange(1905, 2023)
    yearview = yearview.reindex(new_index, fill_value=0)

    #create bar plot
    plt.clf()
    plt.bar(x = yearview.index, height = yearview.total, color='r', alpha=1, label='Failed attempts')
    plt.bar(x = yearview.index, height = yearview.successes, color='k', alpha=1, label='Successful attempts')
    plt.legend()
    plt.title('Ascension attempts per year')
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/ascensions_per_year.png', dpi=640)

    #focus on years where more than 3 ascension attempts took place
    cut = yearview.loc[yearview.total > 3]

    X_train = cut.index.values.reshape(-1,1)
    Y_train = cut.ratio.values.reshape(-1,1)
    linmodel = LinearRegression().fit(X_train, Y_train)
    t = np.linspace(cut.index[0], cut.index[-1], 1000)

    #reindex to include missing years
    cut = cut.reindex(np.arange(cut.index[0], cut.index[-1]+1))

    #calculate rolling window average
    rollingwindow = 10
    cut['rollingavg'] = cut.ratio.rolling(rollingwindow, closed='both', center=True).mean()

    plt.clf()
    plt.plot(cut.index, cut.ratio, 'k.')
    plt.plot(cut.index, cut.rollingavg, 'r', label='{} year average'.format(rollingwindow))
    plt.plot(t, linmodel.predict(t.reshape(-1,1)).flatten(), 'g', label='Linear fit, R^2 = {:.2f}'.format(linmodel.score(X_train, Y_train)))
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.title('Success ratio per year')
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/success_ratio_per_year.png', dpi=640)

def EDA_members(exp, plotpath = None):
    '''perform EDA to examine correlations between ascension success and team sizes.'''
    #Create a view comparing successful and successful expeditions per membercount and hired personel count
    memberview = exp.groupby('totmembers').success.agg(['sum', 'count'])
    memberview = memberview.rename(columns={'sum': 'successes', 'count': 'total'})
    memberview['ratio'] = memberview.successes / memberview.total

    hiredview = exp.groupby('totmembers').success.agg(['sum', 'count'])
    hiredview = hiredview.rename(columns={'sum': 'successes', 'count': 'total'})
    hiredview['ratio'] = hiredview.successes / hiredview.total

    #create bar plot
    plt.clf()
    plt.bar(memberview.index, memberview.total, color='r', label = 'Failed attempts')
    plt.bar(memberview.index, memberview.successes, color='k', label = 'Successful attempts')
    plt.title('Ascension attempts per expedition member count')
    plt.xlabel('Expedition members (no hired personel)')
    plt.legend()
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/ascensions_per_member_count.png', dpi=640)

    #create bar plot
    plt.clf()
    plt.bar(hiredview.index, hiredview.total, color='r', label = 'Failed attempts')
    plt.bar(hiredview.index, hiredview.successes, color='k', label = 'Successful attempts')
    plt.title('Ascension attempts per hired personel count')
    plt.xlabel('Hired personel')
    plt.legend()
    if plotpath == None:
        plt.show()
    else:
        plt.savefig(plotpath+'/ascensions_per_hired_count.png', dpi=640)
    

if __name__ == '__main__':
    #read in data
    exp = pd.read_csv('data/expeditions.csv', index_col=0, usecols=[0,1,2,3,4,13,14,15,16,21,31,34,35,36,37,38,39,41,42,44,45,46,48,49,50])

    #drop entries of unrecognised expeditions
    exp = exp.drop(np.argwhere(exp.claimed.values == True).flatten(), axis='rows')
    exp = exp.drop('claimed', axis='columns')

    #compile multiple success field into one - count number of successful routes
    exp['success'] = exp[['success1', 'success2', 'success3', 'success4']].sum(axis=1).map(lambda x: True if x > 0 else False)
    exp = exp.drop(['success1', 'success2', 'success3', 'success4'], axis='columns')

    #drop entries where the amount of hired members is missing
    exp = exp.drop(exp.index[(exp.tothired == 0) & (exp.nohired == False)].tolist(), axis='rows')
    exp = exp.drop('nohired', axis='columns')

    #drop entries where the amount of expedition members is set to 0
    exp = exp.drop(exp.index[exp.totmembers == 0].tolist(), axis='rows')

    EDA_year(exp)
    EDA_peaks(exp)
    EDA_members(exp)