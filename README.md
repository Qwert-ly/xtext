# 说明
## 语料库与数据获取
语料库体积较大，已迁移至[HuggingFace](https://huggingface.co/datasets/Nulll-Official/ctext)托管
- `download_data.py`：自动从HuggingFace仓库下载最新的`.parquet`格式语料库
  - `ctext - all - slice`：自主收集的文言文繁体语料库，主要爬取自[维基文库](http://zh.qikisource.org)、[漢川草廬](http://www.sidneyluo.net)等处
  - `ctext - 副本 - 副本`：`ctext - all - slice` 的先秦部分
  - `ctext - 白话`：自主收集的白话及现代汉语繁体语料库，主要爬取自[维基百科](http://zh.wikipedia.org)、[BWIKI](http://wiki.biligame.com)、[维基文库](http://zh.qikisource.org)、[知乎](http://www.zhihu.com)、[繁體中文書庫](http://www.bwsk.net)等处

## 上古汉语音节表维护与查询工具
[上古汉语音节表](https://zhuanlan.zhihu.com/p/12987993957)是新最小上古汉语(NOCM)成果的记录与直接反映。
- `_maintain.py`：上古汉语音节表的维护与查询工具，另外包含文本用字差异比较、切韵(藤原复原本)查询（数据来自[qieyun-restored](http://github.com/nk2028/qieyun-restored)）、形声考与语料查询等小功能
- `xlsx2json.py`：解析字典表和小韵表，并导出为json
- `ai_check.py`：用DeepSeek API自动化校对音节表内容，支持断点续传
- `json2xlsx.py`：将AI校对结果转换回xlsx
- `形聲考_240425`：《廣韻》形聲考（Sliark再整理, 2024.4.25），在`_maintain.py`中用于展示中古音韵地位

## NLP工具
- `hmm_seg.py`：文言分词器。使用PMI和信息熵发现文言文词汇，生成训练语料，并基于隐马尔可夫模型（HMM）训练古文分词器
- `clust_lda.py`：对文本进行隐含狄利克雷分布（Latent Dirichlet Allocation）软聚类，提取主题代表字，并可视化各篇目的主题概率分布
- `Word2vec.py`：用`gensim.models.Word2Vec()`提取简单nlp数据，附使用示例
- `clust.py`：基于字频向量进行KMeans聚类；由轮廓分数(Silhouette Score)确定最佳聚类数
- `clust4_clear.py`：基于字频向量，创建[余弦相似度](http://zhuanlan.zhihu.com/p/43396514)矩阵并保存；进行层次聚类，结果将保存为html
- `Word2vec.py`：使用`gensim`提取并训练Word2Vec词向量模型
- `util.py`：工具性函数

## 爬虫工具
- `sidneyluo-scraper.py`：用于[漢川草廬](http://www.sidneyluo.net/index.php)的爬虫，可能需要另按页面格式调整
- `wikipedia-scraper.py`：用于[维基百科](http://zh.wikipedia.org)的爬虫
- `wikisource-scraper(2).py`：用于[维基文库](http://zh.qikisource.org)的爬虫
- `zitool-scraper.py`：用于[字统网](http://zi.tools)字源信息的爬虫

## 其他语料库
### github
- [殆知阁古代文献](http://github.com/garychowcmu/daizhigev20)
- [近代汉语语料库数据集(蒋彦廷, 潘雨婷, 杨乐. 基于统计与词嵌入的近代汉语动量结构研究. 2020)](http://github.com/JiangYanting/Pre-modern_Chinese_corpus_dataset)
- 

### 其他
- [BCC语料库](http://bcc.blcu.edu.cn/zh/cid/5)
- [CCL语料库](http://ccl.pku.edu.cn:8080/ccl_corpus)
- 
