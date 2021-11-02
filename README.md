## 电影推荐系统

### 介绍

本项目为本科实习期间做的一个简易的电影推荐系统，基于协同过滤模型，隐语义模型进行推荐，并使用了 `Flask` 框架进行前端开发，包含登录部分。



### 数据

所有数据存放在 `data` 文件夹中。电影和用户的基础数据为 `ml-latest-small` 数据集，内含`movies.csv`  、`ratings.csv` 、`tags.csv` 、`links.csv` ，共610条用户数据， 其来自[MovieLens](http://movielens.org)，具体内容可参见 `data` 文件夹中的 `README.md` 。

`LFM_p.csv` 和 `LFM_q.csv` 为隐语义模型生成的关于电影和观众的概率，训练时间非常长。

`user.csv` 为用户名、密码等信息，自动生成，密码为 `aaaaaa`。

### 运行

直接运行 `run.py` 文件即可本地预览。

展示了110条推荐数据，前10条为统计热度，中间50条为协同过滤最后50条为隐语义。

![](https://bu.dusays.com/2021/07/07/e956178f476be.png)

