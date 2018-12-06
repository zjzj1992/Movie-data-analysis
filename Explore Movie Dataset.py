
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[87]:


#导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#魔法函数
get_ipython().run_line_magic('matplotlib', 'inline')

#读取电影数据
movie_data = pd.read_csv('C:/Users/zhang/Desktop/tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[88]:


#数据表的行列
display(np.shape(movie_data))

#获取头几行的数据
display(movie_data.head())

#获取最后几行的数据
display(movie_data.tail())

#随机获取一些数据
display(movie_data.sample())


# In[89]:


#使用dtype查看各列的数据类型
for column in movie_data.columns:
    print("{}的数据类型是: {}".format(column,np.dtype(movie_data[column])))


# In[90]:


#True是存在缺失值，False是不存在缺失值
print(movie_data.isnull().any())


# In[91]:


movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[92]:


#查看缺失值的数量
print("\n处理之前：\n",movie_data.isnull().sum())

#其中的homepage的缺失值有很多，占了绝大部分，所以该特征可以直接删除，而其他含有缺失值的特征可以用无等统一的字样来填充
print("--------------------------")
#处理数据中的缺失值
movie_data.drop(['homepage'],axis=1,inplace=True)
movie_data.fillna('nothing',inplace=True)

#处理之后数据的变化
print("处理之后：\n",movie_data.isnull().sum())


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[93]:


display(movie_data[['id','popularity','budget','runtime','vote_average']])


# In[94]:


display(movie_data.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,47,48]])


# In[95]:


#读取50—60行的数据
display(movie_data.iloc[49:60])


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[96]:


#获取popularity大于5的所有数据
movie_data.loc[movie_data.popularity > 5]


# In[97]:


#获取popularity大于5且发行年份在1996年之后的所有数据
movie_data.loc[(movie_data.popularity > 5) & (movie_data.release_year > 1996)]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[98]:


#按年份分组，然后求票房的平均值
print(movie_data.groupby('release_year').agg('mean')['revenue'])


# In[99]:


#按导演进行分组，然后降序显示平均知名度
movie_data.groupby('director').agg('mean').sort_values(by='popularity',ascending=False)['popularity']


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[100]:


#这部分内容是因为在生成图的时候总是出现字体的警告信息，所以查资料解决的方案
import matplotlib as mpl
fm = mpl.font_manager
fm.get_cachedir()
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']


# In[101]:


#按popularity进行降序排列，然后获取前20名
top20_popularity = movie_data.sort_values(by='popularity',ascending=False).head(20)
plt.scatter(data=top20_popularity,x = 'original_title',y = 'popularity');
plt.xlabel("Movie")
plt.ylabel("Popularity")
plt.xticks(rotation=90)


# 前三名的电影的popularity远远高于其他电影

# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[105]:


#方式一
movie_data.sort_values(by='release_year',ascending=True,inplace=True)
movie_counts = movie_data.release_year.value_counts()
movie_data_years = movie_data.groupby('release_year').agg('sum')



movie_data_years['mean profit'] = (movie_data_years['revenue'] - movie_data_years['budget'])/movie_counts
release_years = np.arange(1960,2016,1)
plt.errorbar(data=movie_data_years,x = release_years,y = movie_data_years['mean profit'])
plt.xlim(1960,2020);
plt.xlabel('years')
plt.ylabel('Average revenue')


# In[106]:


#方式二
plt.style.use('ggplot')
_, axes = plt.subplots(2, 1, figsize=(10, 10))
movie_data['profit'] = movie_data['revenue'] - movie_data['budget']
target_data = movie_data.groupby('release_year')['profit'].agg(['mean', 'sum'])
axes[0].errorbar(target_data.index, target_data['mean']);
axes[1].errorbar(target_data.index, target_data['sum']);

axes[0].set_ylabel('profit_mean')
axes[1].set_ylabel('profit_sum')
axes[1].set_xlabel('release_year')


# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[107]:


#因为有些电影会包含多个导演，所以需要先将导演拆分出来，更方便统计
tmp = movie_data['director'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('director')
#提炼出‘original_title’和‘revenue’数据，然后与tmp合并
movie_data_split = movie_data[['original_title', 'revenue']].join(tmp)
#返回出产最多的前10位导演的索引
directors_top10 = tmp.value_counts().nlargest(10).index
#筛选出前10位导演，然后按照票房进行排序
target_data = movie_data_split[movie_data_split['director'].isin(directors_top10)].sort_values('revenue', ascending=False)
#作图，生成前10位导演票房最好的三部电影
_, axes = plt.subplots(2, 5, figsize=(15, 12), sharey=True, tight_layout=True)
for key, ax in zip(directors_top10, axes.flat):
    target_data[target_data['director']==key].set_index('original_title')[:3].plot(kind='bar', ax=ax, legend=False) 
    ax.set_ylabel('reveune')
    ax.set_title(key)


# 导演Steven Spielberg的票房是最好的，最低票房也比其他导演要好；另外Woody Allen和Brian De Palma导演的电影总体不好，处于低票房的位置

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[108]:


#基础色
base_color = sb.color_palette()[0]
#筛选年份
movie_year = movie_data['release_year'].between(1968,2015)

#筛选月份
movie_june = pd.to_datetime(movie_data['release_date']).dt.month == 6

#根据条件作图
plt.figure(figsize=[18,5])
sb.countplot(data=movie_data[movie_year&movie_june],x='release_year',color=base_color)


# 从1968到1985年总体成上升趋势，之后开始下滑，直到1992年开始才开始有回升的趋势，然后从大概2000年开始，上升势头明显起来

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[109]:


#将‘genres’拆分出来
movie_genres = movie_data.drop('genres',axis=1).join(movie_data['genres'].str.split('|',expand=True).stack().reset_index(level=1,drop=True).rename('genres'))
#筛选年份
movie_years = movie_genres['release_year'].between(1968,2015)
#筛选月份
movie_june = pd.to_datetime(movie_genres['release_date']).dt.month == 6
#筛选电影类型
movie_genre = movie_genres['genres'].isin(['Drama','Comedy'])
#利用三个条件作图
plt.figure(figsize=[18,5])
sb.countplot(data=movie_genres[movie_years&movie_june&movie_genre],x='release_year',hue='genres');
plt.xticks(rotation=90);


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
