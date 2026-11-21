# 说明
## 语料库与数据获取
语料库体积较大，已迁移至[HuggingFace](https://huggingface.co/datasets/Nulll-Official/ctext)托管
- `download_data.py`：自动从HuggingFace仓库下载最新的`.parquet`格式语料库
  - `ctext - all - slice`：自主收集的文言文繁体语料库，主要爬取自[维基文库](http://zh.qikisource.org)、[漢川草廬](http://www.sidneyluo.net)等处
  - `ctext - 副本 - 副本`：`ctext - all - slice` 的先秦部分
  - `ctext - 白话`：自主收集的白话及现代汉语繁体语料库，主要爬取自[维基百科](http://zh.wikipedia.org)、[BWIKI](http://wiki.biligame.com)、[维基文库](http://zh.qikisource.org)、[知乎](http://www.zhihu.com)、[繁體中文書庫](http://www.bwsk.net)等处

## 上古汉语音节表维护与查询工具
[上古汉语音节表](https://zhuanlan.zhihu.com/p/12987993957)是基于文献的上古汉语（Philology-Based Old Chinese, PBOC）成果的记录与直接反映。包含音节表、字典表、小韵表3张表，音节表可用于观察声韵搭配，字典表用于记录字词释义、语文学考释、[线上查询](https://qwert-ly.github.io/xtext/)，小韵表用于[汉字音典APP](https://github.com/osfans/MCPDict)查询

- `_maintain.py`：上古汉语音节表的维护与查询工具，另外包含文本用字差异比较、切韵(藤田拓海复原本)查询（数据来自[qieyun-restored](http://github.com/nk2028/qieyun-restored)）、形声考与语料查询等小功能
- `xlsx2json.py`：解析字典表和小韵表，并导出为json
- `ai_check.py`：用DeepSeek API自动化校对字典表内容，支持断点续传
- `json2xlsx.py`：将AI校对结果转换回xlsx
- `形聲考_240425`：《廣韻》形聲考（Sliark再整理, 2024.4.25），在`_maintain.py`中用于展示中古音韵地位
- `音節表2小韻表.cls`：将音节表转换成小韵表的VBA脚本
- `docs/index.html`：上古汉语音节表的查询界面

## NLP工具
- `hmm_seg.py`：文言分词器。使用PMI和信息熵发现文言文词汇，生成训练语料，并基于隐马尔可夫模型（HMM）训练古文分词器
- `clust_lda.py`：提取文本词频矩阵，进行隐含狄利克雷分布（Latent Dirichlet Allocation）软聚类
- `clust_nmf.py`：提取文本TF-IDF矩阵（高频词过滤），进行非负矩阵分解（NMF）软聚类
- `clust.py`：基于字频向量进行KMeans聚类；由轮廓分数(Silhouette Score)确定最佳聚类数
- `clust4_clear.py`：基于字频向量，创建[余弦相似度](http://zhuanlan.zhihu.com/p/43396514)矩阵并保存；进行层次聚类，结果将保存为html
- `Word2vec.py`：用`gensim.models.Word2Vec()`提取并训练Word2Vec词向量模型
- `util.py`：工具性函数

## 爬虫工具
- `sidneyluo-scraper.py`：用于[漢川草廬](http://www.sidneyluo.net/index.php)的爬虫，可能需要另按页面格式调整
- `wikipedia-scraper.py`：用于[维基百科](http://zh.wikipedia.org)的爬虫
- `wikisource-scraper(2).py`：用于[维基文库](http://zh.wikisource.org)的爬虫
- `zitool-scraper.py`：用于[字统网](http://zi.tools)字源信息的爬虫

## 其他语料库
### github
- [殆知阁古代文献](http://github.com/garychowcmu/daizhigev20)
- [近代汉语语料库数据集(蒋彦廷, 潘雨婷, 杨乐. 基于统计与词嵌入的近代汉语动量结构研究. 2020)](http://github.com/JiangYanting/Pre-modern_Chinese_corpus_dataset)

### 其他
- [BCC语料库](http://bcc.blcu.edu.cn/zh/cid/5)
- [CCL语料库](http://ccl.pku.edu.cn:8080/ccl_corpus)
- 

---

## 数据说明
PBOC数据原则上于每月21日更新，提供两种JSON分发形式：
- GitHub Releases中的default.json（推荐下游项目使用）：单文件完整版数据，包含全部字头、拟音、谐声信息、释义与注释。
- docs目录下的网页端分包数据：为网页端加载优化，base.json.gz仅含字头、拟音、谐声域信息（体积小，便于快速查音），extra.json.gz存放释义、注释等其他信息（按需加载）。
> PBOC仍在持续修订中，每月更新均可能涉及**字头拟音修改、收字增删、多音字读音增减、字形替换**及大量释义与注释修订。引用数据时，建议标注所采用的Release版本号（`YY.MM.DD`）。

## 许可证
本仓库采用分类许可（详见[LICENSE](./LICENSE)）：
- PBOC原创数据与说明文档（[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)）：本项目的原创PBOC数据集（含GitHub Releases发布的JSON、docs/base.json.gz、docs/extra.json.gz、docs/data-version.json中的拟音、释义与考释注释）及原创文档均采用CC BY 4.0协议授权。署名“Nulll (PBOC / Philology-Based Old Chinese)”、附上本仓库或查询页链接并标明数据版本号，即视为满足署名要求。维基、词典类项目集中署名即可，无需逐词条重复标注。
- 原创代码与网页前端（[MIT License](https://opensource.org/licenses/MIT)）：本仓库由作者原创的Python脚本、VBA脚本及`docs/`目录下的查询网页前端代码采用MIT License授权。
- 仓库内的`形聲考_240425`、`tshet-uinh`、`tshet-uinh-sr`、第三方字体及外部链接所涉语料库，**均不在**上述开源许可授权范围内，其权利归原权利人所有，使用时请遵循原始来源的许可条件。

