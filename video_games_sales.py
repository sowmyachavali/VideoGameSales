# DSC530
# Final project - videogame_sales.py
# Author Sowmya Chavali
# 03/04/21

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import thinkstats2
import thinkplot

### Sales of video games per genre.
'''
Observations
From the above graphs the following points can be summarized.
World-wide the most popular genre is Action followed by Sports and Shooter.
The most popular genre across North America and Europe is also Action.
The most popular genre in Japan is Role-Playing.
'''
def genre_sales(data_vg,region_arr):
    ##### Sales of video games per genre
    df_genre = data_vg.groupby('Genre')
    for region in region_arr:
        xrange = np.arange(1, len(df_genre.sum()) + 1)
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        df_to_plot = df_genre.sum().sort_values(by=region, ascending=False)[::-1]
        df_to_plot[region].plot(kind='barh')
        # labels
        ax[1].set_ylabel(None)
        ax[1].tick_params(axis='both', which='major', labelsize=13)
        ax[1].set_xlabel(region+' (in millions)', fontsize=15, labelpad=21)
        # spines
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].grid(False)
        # annotations
        for x, y in zip(np.arange(len(df_genre.sum()) + 1),
                    df_genre.sum().sort_values(by=region, ascending=False)[::-1][region]):
            label = "{:}".format(y)
            labelr = round(y, 2)
            plt.annotate(labelr,  # this is the text
                         (y, x),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(6, 0),  # distance from text to points (x,y)
                        ha='left', va="center")

        # donut chart
        theme = plt.get_cmap('Blues')
        ax[0].set_prop_cycle("color", [theme(1. * i / len(df_to_plot)) for i in range(len(df_to_plot))])
        wedges, texts, _ = ax[0].pie(df_to_plot[region], wedgeprops=dict(width=0.45), startangle=-45,
                                     labels=df_to_plot.index,
                                     autopct="%.1f%%", textprops={'fontsize': 13, })
        plt.tight_layout()
        plt.show()

### Sales of video games per platform
'''
Observations
From the above graphs the following points can be summarized.
Globally the most popular platform till date is PS2 followed by Xbox360 and PS3.
The most popular platform in till dateNorth America is Xbox 360 followed by PS2 and Wii
The most popular platform in Europe is PS3 followed by PS2 and Xbox360
The most popular platform in Japan is DS followed by PS and PS2
'''

def platform_sales(data_vg,region_arr):
    df_platform = data_vg.groupby('Platform')
    for region in region_arr:
        df_platform_plot = df_platform.sum().sort_values(by=region, ascending=False).head(12)[::-1]
        xrange = np.arange(1, len(df_platform_plot) + 1)
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        df_platform_plot[region].plot(kind='barh', color='#961515', alpha=.9)
        # labels
        ax[1].set_ylabel(None)
        ax[1].tick_params(axis='both', which='major', labelsize=13)
        ax[1].set_xlabel(region+' (in millions)', fontsize=15, labelpad=21)
        # spines
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].grid(False)
        # annotations
        for x, y in zip(np.arange(len(df_platform_plot) + 1), df_platform_plot[region]):
            label = "{:}".format(y)
            labelr = round(y, 2)
            plt.annotate(labelr,  # this is the text
                         (y, x),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(6, 0),  # distance from text to points (x,y)
                        ha='left', va="center")
        # donut chart
        theme = plt.get_cmap('Reds')
        ax[0].set_prop_cycle("color", [theme(1. * i / len(df_platform_plot)) for i in range(len(df_platform_plot))])
        wedges, texts, _ = ax[0].pie(df_platform_plot[region], wedgeprops=dict(width=0.45), startangle=-45,
                                     labels=df_platform_plot.index,
                                     autopct="%.1f%%", textprops={'fontsize': 13, })
        plt.tight_layout()
        plt.show()

'''
I want to visualize the total number of copies sold by platform and analyze the regions where they were sold.
Having the regions already separated into columns helps a lot; we only need to group the records by 
‘Platform’ and sum the values from NA_Sales to Global_Sales.
'''
def platform_region(data_vg):
    df_grouped = data_vg.groupby('Platform').sum()[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
#Let’s plot a bar for each platform and region and get a look at the result.
    # define figure
    fig, ax = plt.subplots(1, figsize=(10, 5))
    # numerical x
    x = np.arange(0, len(df_grouped.index))
    # plot bars
    plt.bar(x - 0.3, df_grouped['NA_Sales'], width=0.2, color='#1D2F6F')
    plt.bar(x - 0.1, df_grouped['EU_Sales'], width=0.2, color='#8390FA')
    plt.bar(x + 0.1, df_grouped['JP_Sales'], width=0.2, color='#6EAF46')
    plt.bar(x + 0.3, df_grouped['Other_Sales'], width=0.2, color='#FAC748')
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # x y details
    plt.ylabel('Millions of copies')
    plt.xticks(x, df_grouped.index)
    plt.xlim(-0.5, 31)
    # grid lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
    # title and legend
    plt.title('Video Game Sales By Platform and Region', loc='left')
    plt.legend(['NA', 'EU', 'JP', 'Others'], loc='upper left', ncol=4)
    plt.show()

'''
the above chart is hard to read. Let’s try the stacked bar chart and add a few adjustments.
First, we can sort the values before plotting, giving us a better sense of order and making it easier 
to compare the bars. We’ll do so with the ‘Global Sales’ column since it has the total.
This is way more readable than the last one.
The idea here is to compare the platforms' total sales and understand each platform's composition.
'''
def platform_region_stacked(data_vg):
    df_grouped = data_vg.groupby('Platform').sum()[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
    df_grouped = df_grouped.sort_values('Global_Sales')
    fields = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    colors = ['#1D2F6F', '#8390FA', '#6EAF46', '#FAC748']
    labels = ['NA', 'EU', 'JP', 'Others']
    # figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 5))
    # plot bars
    left = len(df_grouped) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df_grouped.index, df_grouped[name], left=left, color=colors[idx])
        left = left + df_grouped[name]
    # title, legend, labels
    plt.title('Video Game Sales By Platform and Region\n', loc='left')
    plt.legend(labels, bbox_to_anchor=([0.55, 1, 0, 0]), ncol=4, frameon=False)
    plt.xlabel('Millions of copies of all games')
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.show()


## Sales of game titles across all regions
def title_sales(data_vg):
    df_game_title = data_vg.groupby('Name').sum()
    df_game_title[['Global_Sales','NA_Sales', 'EU_Sales',
                   'JP_Sales']].sort_values(by='Global_Sales',ascending=
    False).head(12)[::-1].plot(kind='barh', figsize=(10, 5), grid=False)


'''
Top 5 publishers with highest global sales
Here Nintendo takes the top spot
'''
def top5_publishers(data_vg):
    Publisher = list(data_vg.Publisher.unique())
    global_sale_of_every_Publisher = pd.Series(dtype=float)
    for pub in Publisher:
        data = data_vg.loc[data_vg.Publisher == pub]
        global_sale = sum(data.Global_Sales)
        global_sale_of_every_Publisher[pub] = global_sale

    top_5 = global_sale_of_every_Publisher[:5]
    plt.figure(figsize=(8, 5))
    plt.pie(top_5, labels=top_5.index, autopct="%.2f%%", textprops={"fontsize": 13}, labeldistance=1.05)
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Top 5 Publishers", fontdict={"fontsize": 25, "fontweight": 100})
    plt.savefig("Top 5 Publishers", dpi=300)
    plt.show()


# Growth of the top Publisher of games over years
def growth_over_yrs(data_vg):
    Nintendo = data_vg.loc[data_vg.Publisher == "Nintendo"]
    Nintendo_1 = Nintendo.sort_values(by="Year")
    Nintendo_1 = Nintendo_1.dropna()
    Nintendo_years = Nintendo.Year.unique()
    Nintendo_profit_year = pd.Series(dtype=float)
    for yea in Nintendo_years:
        data_of_year = Nintendo_1.loc[Nintendo_1.Year == yea]
        total_of_year = data_of_year.Global_Sales.sum()
        Nintendo_profit_year[yea] = total_of_year
    Nintendo_profit_year = Nintendo_profit_year.sort_index()
    Nintendo_profit_year = Nintendo_profit_year
    plt.plot(Nintendo_profit_year)
    plt.xlabel("Years", size=14)
    plt.ylabel("Unit Sales", size=14)
    plt.title("Nintendo Global Sales from 1983 to 2016", fontdict={"fontsize": 15})
    plt.xticks([i for i in range(1983, 2017, 3)])
    plt.savefig("Nintendo Global Sales from 1983 to 2016", dpi=300)
    plt.show()


# Year Wise Video Game Release Count
'''
For this I wanted to try my hands on plotly which I recently started working on.
The plot says that the year 2009 has the highest number of games released.
'''
def release_by_year(data_vg):
    # Video Game Count by Year
    yearwisegame = data_vg.groupby('Year')['Name'].count().reset_index()
    # Yearwise Total Game Published
    fig = go.Figure(go.Bar(x=yearwisegame['Year'], y=yearwisegame['Name'],
                           marker={'color': yearwisegame['Name'], 'colorscale': 'Viridis'}))
    fig.update_layout(title_text='Video Game Release by Year', xaxis_title="Year",
                      yaxis_title="Number of Games Released")
    fig.show()


# global sales in millions
'''
 As shown in the figure, most of the sales fall within the value range of 10. We will use that
 information with the next plots.
'''
def global_sales(data_vg):
    fig = px.box(data_vg, y="Global_Sales", points="all", height=400)
    fig.show()


## Changes in sales over time
'''
 As we can see from the plot, after a small dip in sales 2003/2004,sales exploded and peaked
 2008-2010. Also known as the golden age of gaming. The sales volume in a year is often proportional
 of the games released in the given year. However, one outlier being 2004, where in my opinion, the
 resale/collectivity of games began to gain traction.
'''
def sales_over_time(data_vg):
    ## Total volume of sales

    df_yearcount = data_vg.groupby(data_vg['Year'])[['Rank']].count().rename(columns={'Rank': 'counts'})
    df_yearsales = data_vg.groupby(data_vg['Year'])[['Global_Sales']].sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df_yearcount.index, y=df_yearcount['counts'], marker=dict(color='rgba(17, 145, 171, 0.6)'),
               name='counts'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_yearsales.index, y=df_yearsales['Global_Sales'], name='Global_Sales'),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="<b>Sales in Millions</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Global_Sales</b>", secondary_y=True)
    fig.show()

def main():
    ### Importing data

    data_vg = pd.read_csv("/Users/raghuveeryellapantula/Desktop/vgsales.csv")
    # setting to display to show all the columns on screen
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 11)

    ### Lets take a quick overview of the data
    print(data_vg.head())
    # database dimension
    print("Database dimension     :", data_vg.shape)
    print("Database size          :", data_vg.size)
    print(data_vg.info())
    # checking numerical columns statistics
    print(data_vg.describe())

    ##### Data cleaning

    print("Count of null values:-\n", data_vg.isnull().sum())
    '''
    Counting the missing values.
    We have 16598 records in our data. Out of that 271 records are missing from Year and 
    58 records from Publisher.
    We will drop these records because removing them won't distrub our EDA.
    '''
    data_vg = data_vg.dropna(axis=0, inplace=False)
    print("Count of null values:-\n", data_vg.isnull().sum())

    # Handling the year variable
    year_data = data_vg['Year']
    print("Max Year Value: ", year_data.max())

    '''Insight:
        This is an anomoly as the downloaded data is till the year 2016. 
        We will remove the row(s) with wrong year or we will try to find the real year for those columns.
    '''
    print("data of years post 2016 \n",data_vg[data_vg["Year"] > 2017.0])

    data_vg = data_vg[data_vg["Year"] < 2017]
    print(data_vg.head())

    # Generating a correlation table to observe the iteraction between variables.
    print(data_vg.corr().round(1))

    #hist
    plt.hist(data_vg['Year'])
    plt.title('Year')
    plt.show()
    plt.hist(data_vg['NA_Sales'],bins=50, range=[0,3])
    plt.title('NA Sales')
    plt.show()
    plt.hist(data_vg['EU_Sales'],bins=50, range=[0,2])
    plt.title('EU Sales')
    plt.show()
    plt.hist(data_vg['JP_Sales'],bins=50, range=[0,2])
    plt.title('JP Sales')
    plt.show()
    plt.hist(data_vg['Global_Sales'],bins=50, range=[0,20])
    plt.title('Global Sales')
    plt.show()

    cdf = thinkstats2.Cdf(data_vg.Global_Sales, label='global sales')
    thinkplot.Cdf(cdf)
    thinkplot.Config(xlabel='Global sales in millions', ylabel='CDF', loc='upper left')
    plt.show()

    ## scatter plots
    plt.scatter(data_vg.Year,data_vg.Global_Sales)
    plt.title('Year vs Global Sales(in millions)')
    plt.show()
    plt.scatter(data_vg.Year, data_vg.NA_Sales)
    plt.title('Year vs NA Sales(in millions)')
    plt.show()
    plt.scatter(data_vg.Year, data_vg.EU_Sales)
    plt.title('Year vs EU Sales(in millions)')
    plt.show()
    plt.scatter(data_vg.Year, data_vg.JP_Sales)
    plt.title('Year vs JP Sales(in millions)')
    plt.show()

    region_sales = np.array(['Global_Sales','EU_Sales','NA_Sales','JP_Sales'])

# Sales of video games per genre
    genre_sales(data_vg,region_sales)

# Sales of video games per platform
    platform_sales(data_vg,region_sales)

### Best-selling game titles till date
# Sales of game titles across all regions
    title_sales(data_vg)

# platform and region
    platform_region(data_vg)
    platform_region_stacked(data_vg)

# top5 publishers
    top5_publishers(data_vg)

# growth of top publisher Nintendo over years.
    growth_over_yrs(data_vg)

# Year Wise Video Game Release Count
    release_by_year(data_vg)

#  global sales in millions
    global_sales(data_vg)

# sales over time
    sales_over_time(data_vg)

if __name__ == "__main__":
    main()

