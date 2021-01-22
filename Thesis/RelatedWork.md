# **Related Work**

<span id="return"> </span>
> ## **CONTENT** 

- [Time Series Representation/Classification/Forecasting](#tsrcf)
    - **Part 1:** Distance/Feature/Kernel-based
    - [Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles](#anthonybagnall2015)
    - [Similarity Measure Selection for Clustering Time Series Databases](#usuemori2016)
    - [Mining Novel Multivariate Relationships In Time Series Data Using Correlation Networks](#saurabhagrawal2020)
    - [Classifying Time Series Using Local Descriptors with Hybrid Sampling](#jiapingzhao2016)
    - [Random Warping Series: A Random Features Method for Time-Series Embedding](#lingfeiwu2018)
    - [A Review on Distance Based Time Series Classification](#amaiaabanda2019)
    - [A Fast Shapelet Selection Algorithm For Time Series Classification](#cunji2019)
    - [Unsupervised Classification of Multivariate Time Series Using VPCA and Fuzzy Clustering with Spatial Weighted Matrix Distance](#honghe)   
    - [Highly Comparative Feature-Based Time-Series Classification](#bendfulcher)
    - [Efficient Temporal Pattern Recognition by Means of Dissimilarity Space Embedding With Discriminative Prototypes](#briankenjiiwana)
    - [A Global Averaging Method For Dynamic Time Warping, With Applications To Clustering](#francois2011)
    - **Part 2:** Model-based ((UN)supervised / deep learning)
    - [Time Series Classification with Multivariate Convolutional Neural Network](#chienliangliu2019)
    - [Learning Representations for Time Series Clustering](#qianlima)
    - [Multivariate LSTM-FCNs for time series classification](#fazlekarim)
    - [Temporal representation learning for time series classification](#yupenghu)
    - [Deep Temporal Clustering : Fully Unsupervised Learning of Time-Domain Features](#naveensai)

- [Anomaly Detection](#anomaly)
    - [A Deep Neural Network For Unsupervised Anomaly Detection And Diagnosis In Multivariate Time Series Data](#chuxuzhang)

- [Clustering Analysis](#cluster)
    - [Structural Deep Clustering Network](#deyubo)
    - [Unsupervised deep embedding for clustering analysis](#junyuanxie)

- [Machine Learning](#ml)
    - [LightGBM: A highly efficient gradient boosting decision tree](#guolinke)

- [Feature Interaction/Selection](#featureis)

- [Computer Vision (CV)](#cv)
    - **Part 1:** Image Classification/Detection/Applications/Networks
    - [ImageNet Classification With Deep Convolutional Neural Networks](#alex2012)
    - [Temporal Convolutional Networks for Action Segmentation and Detection](#colin2017)
    - [Network in Network](#minlin2014)
    - [Non-local Neural Networks](#xiaolongwang2018)
    - [Gradient-Based Learning Applied to Document Recognition](#lecun1998)
    - [Visualizing and Understanding Convolutional Networks](#matthew2014)
    - [Fully Convolutional Networks for Semantic Segmentation](#evan2017)
    - [Very deep convolutional networks for large-scale image recognition](#karen2015)
    - []()
    - **Part 2:**  Image Representation Learning ((Dis)entangled)
    - [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](#francesco2019)
    - [Disentangling by Factorising ](#hyunjik2018)
    - [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](#xichen2016)
    - [Isolating Sources of Disentanglement in VAEs](#ricky2018)
    - [Neural Discrete Representation Learning](#aaronvanden2017)
    - [β-VAE: Learning Basic Visual Concepts With A Constrained Variational Framework](#irina2017)
    - [Understanding disentangling in β-VAE](#christopher2018)

- [Natural Language Processing (NLP)](#nlp)
    - (1) Languages Classification/Applications/Networks
    - [Long Short-Term Memory](#sepp1997)
    - [Dilated Recurrent Neural Networks](#shiyuchang2017)
    - [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](#shaojiebai)
    - (2) Words Representation Learning

- [Recommender Systems](#rs)

- [Fault Diagnosis and Detection](#fdd)
    - [Data-Driven Fault Diagnosis Method Based on Compressed Sensing and Improved Multiscale Network](#zhongxuhu2020)
    - [Deep Convolutional Neural Network Model based Chemical Process Fault Diagnosis](#haowu)

- [Valve Stiction](#valve)



----
<span id="tsrcf"> </span>
> ## **Time Series Representation/Classification/Forecasting**
<span id="dfk"> </span>
>> ### Distance/Feature/Kernel-based  
<span id="anthonybagnall2015"> </span> 
- *Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles*  [^^^](#return)
    - https://ieeexplore.ieee.org/document/7069254
    - Anthony Bagnall, Jason Lines, Jon Hills, and Aaron Bostrom
    - IEEE Transactions on Knowledge and Data Engineering. 2015, 27(9): 2522-2535
    - Recently, two ideas have been explored that lead to more accurate algorithms for time-series classification (TSC). First, it has been shown that the simplest way to gain improvement on TSC problems is to transform into an alternative data space where discriminatory features are more easily detected. Second, it was demonstrated that with a single data representation, improved accuracy can be achieved through simple ensemble schemes. We combine these two principles to test the hypothesis that forming a collective of ensembles of classifiers on different data transformations improves the accuracy of time-series classification. The collective contains classifiers constructed in the time, frequency, change, and shapelet transformation domains. For the time domain, we use a set of elastic distance measures. For the other domains, we use a range of standard classifiers. Through extensive experimentation on 72 datasets, including all of the 46 UCR datasets, we demonstrate that the simple collective formed by including all classifiers in one ensemble is significantly more accurate than any of its components and any other previously published TSC algorithm. We investigate alternative hierarchical collective structures and demonstrate the utility of the approach on a new problem involving classifying Caenorhabditis elegans mutant types.
    - 在不同的数据转换上采用集成的分类器将会提高时序数据的分类性能，这篇论文认为或者假设数据的transformations比分类器的选择更重要

<span id="usuemori2016"> </span> 
- *Similarity Measure Selection for Clustering Time Series Databases* [^^^](#return)
    - https://ieeexplore.ieee.org/document/7172543 
    - Usue Mori, Alexander Mendiburu, and Jose A. Lozano
    - IEEE Transactions on Knowledge and Data Engineering. 2016, 28(1), 181-195
    - In the past few years, clustering has become a popular task associated with time series. The choice of a suitable distance measure is crucial to the clustering process and, given the vast number of distance measures for time series available in the literature and their diverse characteristics, this selection is not straightforward. With the objective of simplifying this task, we propose a multi-label classification framework that provides the means to automatically select the most suitable distance measures for clustering a time series database. This classifier is based on a novel collection of characteristics that describe the main features of the time series databases and provide the predictive information necessary to discriminate between a set of distance measures. In order to test the validity of this classifier, we conduct a complete set of experiments using both synthetic and real time series databases and a set of five common distance measures. The positive results obtained by the designed classification framework for various performance measures indicate that the proposed methodology is useful to simplify the process of distance selection in time series clustering tasks.
    - 设定了71个备用的时序特征，以及5种不同的基于距离的时序距离策略，研究分析了哪些特征在哪种距离进行分类时能有好的效果，进行特征的选择。该策略的主要问题是如何能够构建训练集，因为在现有的数据集中没有能够标注特征重要度的数据，具体方法参考论文Section4

<span id="saurabhagrawal2020"> </span> 
- *Mining Novel Multivariate Relationships In Time Series Data Using Correlation Networks* [^^^](#return)
    - https://ieeexplore.ieee.org/document/8693798 
    - Saurabh Agrawal, Michael Steinbach, Daniel Boley, Snigdhansu Chatterjee, Gowtham Atluri, Anh The Dang, Stefan Liess, and Vipin Kumar
    - IEEE Transactions on Knowledge and Data Engineering, 2020, 32(9): 1798-1811
    - In many domains, there is significant interest in capturing novel relationships between time series that represent activities recorded at different nodes of a highly complex system. In this paper, we introduce multipoles, a novel class of linear relationships between more than two time series. A multipole is a set of time series that have strong linear dependence among themselves, with the requirement that each time series makes a significant contribution to the linear dependence. We demonstrate that most interesting multipoles can be identified as cliques of negative correlations in a correlation network. Such cliques are typically rare in a real-world correlation network, which allows us to find almost all multipoles efficiently using a clique-enumeration approach. Using our proposed framework, we demonstrate the utility of multipoles in discovering new physical phenomena in two scientific domains: climate science and neuroscience. In particular, we discovered several multipole relationships that are reproducible in multiple other independent datasets and lead to novel domain insights.
    - 不需要label无监督特征工程方式，提出一种称为multipoles的概念，用于寻找多维时序数据的最大线性相关集合，可以用来发现时序数据中存在的模式，捕获复杂的信号关系，考虑在不同的模式中，出现multipoles的数量会不一样或者有不同的表征方式，本质上这是一种工具，用于发现时序中存在的模式。

<soan id="jiapingzhao2016"> </span>
- *Classifying Time Series Using Local Descriptors with Hybrid Sampling* [^^^](#return)
    - https://ieeexplore.ieee.org/document/7300428 
    - Jiaping Zhao, Laurent Itti
    - IEEE Transactions on Knowledge and Data Engineering. 2016, 28(3): 623-637
    - Time series classification (TSC) arises in many fields and has a wide range of applications. Here, we adopt the bag-of-words (BoW) framework to classify time series. Our algorithm first samples local subsequences from time series at feature-point locations when available. It then builds local descriptors, and models their distribution by Gaussian mixture models (GMM), and at last it computes a Fisher Vector (FV) to encode each time series. The encoded FV representations of time series are readily used by existing classifiers, e.g., SVM, for training and prediction. In our work, we focus on detecting better feature points and crafting better local representations, while using existing techniques to learn codebook and encode time series. Specifically, we develop an efficient and effective peak and valley detection algorithm from real-case time series data. Subsequences are sampled from these peaks and valleys, instead of sampled randomly or uniformly as was done previously. Then, two local descriptors, Histogram of Oriented Gradients (HOG-1D) and Dynamic time warping-Multidimensional scaling (DTW-MDS), are designed to represent sampled subsequences. Both descriptors complement each other, and their fused representation is shown to be more descriptive than individual ones. We test our approach extensively on 43 UCR time series datasets, and obtain significantly improved classification accuracies over existing approaches, including NNDTW and shapelet transform.
    - 这篇论文主要是为了获得更好的时序局部特征，提出两种局部特征表述器，Histogram of Oriented Gradients (HOG) of 1D time series 和 Dynamic time warping-multidimen- sional scaling (DTW-MDS)，两种特征描述器相互补充，提供更好的表征

<span id="lingfeiwu2018"> </span>
- *Random Warping Series: A Random Features Method for Time-Series Embedding* [^^^](#return)
    - https://arxiv.org/abs/1809.05259v1 
    - https://github.com/IBM/RandomWarpingSeries 
    - Lingfei Wu, Ian En-Hsu Yen, Jinfeng Yi, Fangli Xu, Qi Lei, Michael J. Witbrock
    - arXiv, 2018
    - Time series data analytics has been a problem of substantial interests for decades, and Dynamic Time Warping (DTW) has been the most widely adopted technique to measure dissimilarity between time series. A number of global-alignment kernels have since been proposed in the spirit of DTW to extend its use to kernel-based estimation method such as support vector machine. However, those kernels suffer from diagonal dominance of the Gram matrix and a quadratic complexity w.r.t. the sample size. In this work, we study a family of alignment-aware positive definite (p.d.) kernels, with its feature embedding given by a distribution of \emph{Random Warping Series (RWS)}. The proposed kernel does not suffer from the issue of diagonal dominance while naturally enjoys a \emph{Random Features} (RF) approximation, which reduces the computational complexity of existing DTW-based techniques from quadratic to linear in terms of both the number and the length of time-series. We also study the convergence of the RF approximation for the domain of time series of unbounded length. Our extensive experiments on 16 benchmark datasets demonstrate that RWS outperforms or matches state-of-the-art classification and clustering methods in both accuracy and computational time.
    - Problems: Unfortunately, the DTW distance does not correspond to a valid positive- definite (p.d.) kernel and thus direct use of DTW leads to an indefinite kernel matrix that neither corresponds to a loss minimization problem nor giving a convex optimization problem. 因此提出一种Random Warping Series （RWS） Approximation主要是为了解决一般通过距离度量来构造核时，矩阵的非半正定问题。是否考虑用深度网络来代替核的作用，避免这些问题。

<span id="amaiaabanda2019"> </span>
- *A Review on Distance Based Time Series Classification* [^^^](#return)
    - https://link.springer.com/article/10.1007/s10618-018-0596-4 
    - Amaia Abanda, Usue Mori, Jose A. Lozano
    - Data Mining and Knowledge Discovery volume 33, pages: 378–412(2019)
    - Time series classification is an increasing research topic due to the vast amount of time series data that is being created over a wide variety of fields. The particularity of the data makes it a challenging task and different approaches have been taken, including the distance based approach. 1-NN has been a widely used method within distance based time series classification due to its simplicity but still good performance. However, its supremacy may be attributed to being able to use specific distances for time series within the classification process and not to the classifier itself. With the aim of exploiting these distances within more complex classifiers, new approaches have arisen in the past few years that are competitive or which outperform the 1-NN based approaches. In some cases, these new methods use the distance measure to transform the series into feature vectors, bridging the gap between time series and traditional classifiers. In other cases, the distances are employed to obtain a time series kernel and enable the use of kernel methods for time series classification. One of the main challenges is that a kernel function must be positive semi-definite, a matter that is also addressed within this review. The presented review includes a taxonomy of all those methods that aim to classify time series using a distance based approach, as well as a discussion of the strengths and weaknesses of each method.
    - 讨论基于距离的时序分类方法，同时进行了优缺点的分析。将时序分类分为feature based (transformed into feature vectors and then classified by a conventional classifier, such as discrete Fourier transform, discrete wavelet), model based, distance based(a (dis)similarity measure between series),这篇论文重点是对基于距离的方法进行了总结，大致分为几类，第一类直接利用全局距离（Global distance features）或局部距离(Local distance features)，嵌入特征（Embedded features： 不直接利用距离度量，而是利用距离的度量生成其他新的表征），基于核方法（Distance kernels， 基于距离度量来生成时序核，这种方法还需再自己看看，不是很熟悉）

<span id="cunji2019"> </span>
- *A Fast Shapelet Selection Algorithm For Time Series Classification* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S1389128618312970
    - Cun Ji, Chao Zhao, Shijun Liu, Chenglei Yang, Li Pan, Lei Wu, Xiangxu Meng
    - Computer Networks, Volume 148, 15 January 2019, Pages 231-240
    - Time series classification has attracted significant interest over the past decade. One of the promising approaches is shapelet based algorithms, which are interpretable, more accurate and faster than most classifiers. However, the training time of shapelet based algorithms is high, even though it is computed off-line. To overcome this problem, in this paper, we propose a fast shapelet selection algorithm (FSS), which sharply reduces the time consumption of shapelet selection. In our algorithm, we first sample some time series from a training dataset with the help of a subclass splitting method. Then FSS identifies the local farthest deviation points (LFDPs) for the sampled time series and selects the subsequences between two nonadjacent LFDPs as shapelet candidates. Using these two steps, the number of shapelet candidates is greatly reduced, which leads to an obvious reduction in time complexity. Unlike other methods that accelerate the shapelet selection process at the expense of a reduction in accuracy, the experimental results demonstrate that FSS is thousands of times faster than the original shapelet transformation method, with no reduction in accuracy. Our results also demonstrate that our method is the fastest among shapelet based methods that have the leading level of accuracy.
    - 提出一种shapelets candidates 的选择方法，不同于其他降低shapelets-based 算法，这篇文章主要是为了减少candidates的数量，而不是降低评估candidates的时间。该策略分为两个步骤，第一步是对时序数据进行采样，也就是采样一些典型的时序序列，从这些序列中进行shapelet的辨识。第二部是提取shapelet候选，包括step1: 辨识endpoints(LFDPs: the point is in subsequence S that has the maximum weight and the point is a maximum dis- tance from the fitting line of S), step2: 通过辨识的LFDPs, 生成Candidates.

<span id="honghe"> </span>
- *Unsupervised Classification of Multivariate Time Series Using VPCA and Fuzzy Clustering with Spatial Weighted Matrix Distance* [^^^](#return)
    - https://ieeexplore.ieee.org/document/8573123
    - Hong He; Yonghong Tan
    - IEEE Transactions on Cybernetics, 2020, Volume: 50, Issue: 3, 1096 - 1105
    - Due to high dimensionality and multiple variables, unsupervised classification of multivariate time series (MTS) involves more challenging problems than those of univariate ones. Unlike the vectorization of a feature matrix in traditional clustering algorithms, an unsupervised pattern recognition scheme based on matrix data is proposed for MTS samples in this paper. To reduce the computational load and time consumption, a novel variable-based principal component analysis (VPCA) is first devised for the dimensionality reduction of MTS samples. Afterward, a spatial weighted matrix distancebased fuzzy clustering (SWMDFC) algorithm is proposed to directly group MTS samples into clusters as well as preserve the structure of the data matrix. The spatial weighted matrix distance (SWMD) integrates the spatial dimensionality difference of elements of data into the distance of MST pairs. In terms of the SWMD, the MTS samples are clustered without vectorization in the dimensionality-reduced feature matrix space. Finally, three open-access datasets are utilized for the validation of the proposed unsupervised classification scheme. The results show that the VPCA can capture more features of MTS data than principal component analysis (PCA) and 2-D PCA. Furthermore, the clustering performance of SWMDFC is superior to that of fuzzy c-means clustering algorithms based on the Euclidean distance or image Euclidean distance.
    - 提出一种基于矩阵数据的无监督识别方案。基于Variable PCA实现对原始时序数据的表征，利用Spatial-Weighted Matrix Distance 来度量不同表征的之间的距离，从而基于fuzzy实现无监督的分类。VPCA没有在特征维度上进行降维，而是在数据长度上进行了降维。SWMD是一种度量矩阵间距离的方式

<span id="bendfulcher"> </span>
- *Highly Comparative Feature-Based Time-Series Classification* [^^^](#return)
    - https://ieeexplore.ieee.org/document/6786425?arnumber=6786425&tag=1
    - Ben D. Fulcher; Nick S. Jones
    - IEEE Transactions on Knowledge and Data Engineering ( Volume: 26, Issue: 12, Dec. 1 2014), 3026 - 3037
    - A highly comparative, feature-based approach to time series classification is introduced that uses an extensive database of algorithms to extract thousands of interpretable features from time series. These features are derived from across the scientific time-series analysis literature, and include summaries of time series in terms of their correlation structure, distribution, entropy, stationarity, scaling properties, and fits to a range of time-series models. After computing thousands of features for each time series in a training set, those that are most informative of the class structure are selected using greedy forward feature selection with a linear classifier. The resulting feature-based classifiers automatically learn the differences between classes using a reduced number of time-series properties, and circumvent the need to calculate distances between time series. Representing time series in this way results in orders of magnitude of dimensionality reduction, allowing the method to perform well on very large data sets containing long time series or time series of different lengths. For many of the data sets studied, classification performance exceeded that of conventional instance-based classifiers, including one nearest neighbor classifiers using euclidean distances and dynamic time warping and, most importantly, the features selected provide an understanding of the properties of the data set, insight that can guide further scientific investigation.
    - 通过time-series科学文献中的方法，为时序数据提取数千个特征。然后使用带有线性分类器的贪婪前向特征选择来选择最能说明类结构的特征，生成的基于特征的分类器使用减少的时间序列属性数量自动了解类之间的差异，并且无需计算时间序列之间的距离。这篇文章认为所选的特征将有助于理解数据集的特性。
    - 基于多种不同的操作，每个操作输出一个定量值，自动选择从文献中提取的上千个特征。 采用贪婪前向特征选择算法对特征进行降维，给定一个分类器（linear discriminant classifier），先选一个性能最好的，再选择与这个特征组合后性能最好的

<span id="briankenjiiwana"> </span>
- *Efficient Temporal Pattern Recognition by Means of Dissimilarity Space Embedding With Discriminative Prototypes* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S0031320316303739
    - Brian Kenji Iwana, Volkmar Frinken, Kaspar Riesen, Seiichi Uchida
    - Pattern Recognition, Volume 64, April 2017, Pages 268-276.
    - Dissimilarity space embedding (DSE) presents a method of representing data as vectors of dissimilarities. This representation is interesting for its ability to use a dissimilarity measure to embed various patterns (e.g. graph patterns with different topology and temporal patterns with different lengths) into a vector space. The method proposed in this paper uses a dynamic time warping (DTW) based DSE for the purpose of the classification of massive sets of temporal patterns. However, using large data sets introduces the problem of requiring a high computational cost. To address this, we consider a prototype selection approach. A vector space created by DSE offers us the ability to treat its independent dimensions as features allowing for the use of feature selection. The proposed method exploits this and reduces the number of prototypes required for accurate classification. To validate the proposed method we use two-class classification on a data set of handwritten on-line numerical digits. We show that by using DSE with ensemble classification, high accuracy classification is possible with very few prototypes.
    - 基于DTW创建N个P维DSE，然后基于Adaboost选择不相关的prototypes, 也就是选择原始DSE的子集。关于初始的DSE集合的选取，通常是从不同的类中选取Prototypes, 极端情况下，所有的Prototypes均属于一个类
    - Example of the DSE: 选取多个原型（prototypes），给定某一个样本，基于这个样本到每个原型的距离，构建DSE向量

<span id="francois2011"> </span>
- *A Global Averaging Method For Dynamic Time Warping, With Applications To Clustering* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S003132031000453X 
    - Francois Petitjean, Alain Ketterlin, Pierre Gancarski
    - Pattern Recognition, 2011, 44, 678-693
    - Mining sequential data is an old topic that has been revived in the last decade, due to the increasing availability of sequential datasets. Most works in this field are centred on the definition and use of a distance (or, at least, a similarity measure) between sequences of elements. A measure called dynamic time warping (DTW) seems to be currently the most relevant for a large panel of applications. This article is about the use of DTW in data mining algorithms, and focuses on the computation of an average of a set of sequences. Averaging is an essential tool for the analysis of data. For example, the K-means clustering algorithm repeatedly computes such an average, and needs to provide a description of the clusters it forms. Averaging is here a crucial step, which must be sound in order to make algorithms work accurately. When dealing with sequences, especially when sequences are compared with DTW, averaging is not a trivial task.   
    Starting with existing techniques developed around DTW, the article suggests an analysis framework to classify averaging techniques. It then proceeds to study the two major questions lifted by the framework. First, we develop a global technique for averaging a set of sequences. This technique is original in that it avoids using iterative pairwise averaging. It is thus insensitive to ordering effects. Second, we describe a new strategy to reduce the length of the resulting average sequence. This has a favourable impact on performance, but also on the relevance of the results. Both aspects are evaluated on standard datasets, and the evaluation shows that they compare favourably with existing methods. The article ends by describing the use of averaging in clustering. The last section also introduces a new application domain, namely the analysis of satellite image time series, where data mining techniques provide an original approach.
    - 这篇


<span id="model"> </span>
>> ### Model-based ((un/semi)-supervised / deep learning)
<span id="chienliangliu2019"> </span>
- *Time Series Classification with Multivariate Convolutional Neural Network* [^^^](#return)
    - https://ieeexplore.ieee.org/document/8437249 
    - Chien-Liang Liu, Wen-Hoar Hsaio, Yao-Chung Tu
    - IEEE Transactions on Industrial Electronics. 2019, 66(6), 4788-4797
    - Time series classification is an important research topic in machine learning and data mining communities, since time series data exist in many application domains. Recent studies have shown that machine learning algorithms could benefit from good feature representation, explaining why deep learning has achieved breakthrough performance in many tasks. In deep learning, the convolutional neural network (CNN) is one of the most well-known approaches, since it incorporates feature learning and classification task in a unified network architecture. Although CNN has been successfully applied to image and text domains, it is still a challenge to apply CNN to time series data. This paper proposes a tensor scheme along with a novel deep learning architecture called multivariate convolutional neural network (MVCNN) for multivariate time series classification, in which the proposed architecture considers multivariate and lag-feature characteristics. We evaluate our proposed method with the prognostics and health management (PHM) 2015 challenge data, and compare with several algorithms. The experimental results indicate that the proposed method outperforms the other alternatives using the prediction score, which is the evaluation metric used by the PHM Society 2015 data challenge. Besides performance evaluation, we provide detailed analysis about the proposed method.
    - 提出一种多变量卷积框架用于多维时序数据的分类，该框架包含4个部分，Input tensor transformation stage，Univariate convolution stage，Multivariate convolution stage，Fully connected stage。最终的测试数据为PHM 2015 challenge 

<span id="qianlima"> </span>
- *Learning Representations for Time Series Clustering* [^^^](#return)
    - Qianli Ma, Jiawei Zheng, Sen Li, Garrison W. Cottrell
    - https://papers.nips.cc/paper/2019/hash/1359aa933b48b754a2f54adb688bfa77-Abstract.html
    - https://github.com/qianlima-lab/DTCR
    - Part of Advances in Neural Information Processing Systems 32 (NeurIPS 2019)
    - Time series clustering is an essential unsupervised technique in cases when category information is not available. It has been widely applied to genome data, anomaly detection, and in general, in any domain where pattern detection is important. Although feature-based time series clustering methods are robust to noise and outliers, and can reduce the dimensionality of the data, they typically rely on domain knowledge to manually construct high-quality features. Sequence to sequence (seq2seq) models can learn representations from sequence data in an unsupervised manner by designing appropriate learning objectives, such as reconstruction and context prediction. When applying seq2seq to time series clustering, obtaining a representation that effectively represents the temporal dynamics of the sequence, multi-scale features, and good clustering properties remains a challenge. How to best improve the ability of the encoder is still an open question. Here we propose a novel unsupervised temporal representation learning model, named Deep Temporal Clustering Representation (DTCR), which integrates the temporal reconstruction and K-means objective into the seq2seq model. This approach leads to improved cluster structures and thus obtains cluster-specific temporal representations. Also, to enhance the ability of encoder, we propose a fake-sample generation strategy and auxiliary classification task. Experiments conducted on extensive time series datasets show that DTCR is state-of-the-art compared to existing methods. The visualization analysis not only shows the effectiveness of cluster-specific representation but also shows the learning process is robust, even if K-means makes mistakes
    - DTCR integrates temporal reconstruction and the K-means objective into a seq2seq model. 融合重构误差和Kmeans损失到Seq2Seq 模型，提出Deep Tempo- ral Clustering Representation (DTCR)，生成一种面向聚类的temproral represeantaions. 采用随机打乱时间步来构建假样本，通过对真假样本的识别，进一步提高表征能力，总的Loss包含三个部分：重构，分类，聚类，参考公式9。

<span id="fazlekarim"> </span>
- *Multivariate LSTM-FCNs for time series classification* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S0893608019301200
    - Fazle Karim, Somshubra Majumdar, Houshang Darabi, Samuel Harford
    - Neural Networks, Volume 116, August 2019, Pages 237-245
    - Over the past decade, multivariate time series classification has received great attention. We propose transforming the existing univariate time series classification models, the Long Short Term Memory Fully Convolutional Network (LSTM-FCN) and Attention LSTM-FCN (ALSTM-FCN), into a multivariate time series classification model by augmenting the fully convolutional block with a squeeze-and-excitation block to further improve accuracy. Our proposed models outperform most state-of-the-art models while requiring minimum preprocessing. The proposed models work efficiently on various complex multivariate time series classification tasks such as activity recognition or action recognition. Furthermore, the proposed models are highly efficient at test time and small enough to deploy on memory constrained systems.
    - 有监督，提出一种MLSTM-FCN结构，实现多变量时序分类。
    
<span id="yupenghu"> </span>
- *Temporal representation learning for time series classification* [^^^](#return)
    - https://link.springer.com/article/10.1007/s00521-020-05179-w
    - Yupeng Hu, Peng Zhan, Yang Xu, Jia Zhao, Yujun Li, Xueqing Li 
    - Neural Computing and Applications, 2020
    - Recent years have witnessed the exponential growth of time series data as the popularity of sensing devices and development of IoT techniques; time series classification has been considered as one of the most challenging studies in time series data mining, attracting great interest over the last two decades. According to the empirical evidences, temporal representation learning-based time series classification has more superiority of accuracy, efficiency and interpretability as compared to hundreds of existing time series classification methods. However, due to the high time complexity of feature process, the performance of these methods has been severely restricted. In this paper, we first presented an efficient shapelet transformation method to improve the overall efficiency of time series classification, and then, we further developed a novel enhanced recurrent neural network model for deep representation learning to further improve the classification accuracy. Experimental results on typical real-world datasets have justified the superiority of our models over several shallow and deep representation learning competitors.
    - 1.提出一种有效的 shapelet transformation， 设定一段序列的变化点turning points (TP), 计算每个TP的重要度，选取一定比例的TP可以重构时序，TP可以反映整个序列的趋势。 2.将原始序列按照滑动窗口进行分割，再送入Bi-LSTM模型中

<span id="naveensai"> </span>
- *Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features* [^^^](#return)
    - https://arxiv.org/abs/1802.01059
    - Naveen Sai Madiraju, Seid M. Sadat, Dimitry Fisher, Homa Karimabadi
    - arXiv:1802.01059, February 2018
    - Unsupervised learning of time series data, also known as temporal clustering, is a challenging problem in machine learning. Here we propose a novel algorithm, Deep Temporal Clustering (DTC), to naturally integrate dimensionality reduction and temporal clustering into a single end-to-end learning framework, fully unsupervised. The algorithm utilizes an autoencoder for temporal dimensionality reduction and a novel temporal clustering layer for cluster assignment. Then it jointly optimizes the clustering objective and the dimensionality reduction objec tive. Based on requirement and application, the temporal clustering layer can be customized with any temporal similarity metric. Several similarity metrics and state-of-the-art algorithms are considered and compared. To gain insight into temporal features that the network has learned for its clustering, we apply a visualization method that generates a region of interest heatmap for the time series. The viability of the algorithm is demonstrated using time series data from diverse domains, ranging from earthquakes to spacecraft sensor data. In each case, we show that the proposed algorithm outperforms traditional methods. The superior performance is attributed to the fully integrated temporal dimensionality reduction and clustering criterion.
    - 提出deep temporal clustering (DTC) 将降维和时间聚类融合到一个端到端的框架，完全无监督。利用自编码器实现降维，设计了一个新的聚类层，聚类层可以使用任意的temporal similarity metric，提供了可视化方法。

----
<span id="anomaly"> </span>
> ## **Anomaly Detection**
<span id="chuxuzhang"> </span>
- *A Deep Neural Network For Unsupervised Anomaly Detection And Diagnosis In Multivariate Time Series Data* [^^^](#return)
    - https://ojs.aaai.org//index.php/AAAI/article/view/3942
    - Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, Nitesh V. Chawla
    - The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI}, 2019, 1409—1416.
    - Nowadays, multivariate time series data are increasingly collected in various real world systems, e.g., power plants, wearable devices, etc. Anomaly detection and diagnosis in multivariate time series refer to identifying abnormal status in certain time steps and pinpointing the root causes. Building such a system, however, is challenging since it not only requires to capture the temporal dependency in each time series, but also need encode the inter-correlations between different pairs of time series. In addition, the system should be robust to noise and provide operators with different levels of anomaly scores based upon the severity of different incidents. Despite the fact that a number of unsupervised anomaly detection algorithms have been developed, few of them can jointly address these challenges. In this paper, we propose a Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED), to perform anomaly detection and diagnosis in multivariate time series data. Specifically, MSCRED first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses in different time steps. Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. Finally, based upon the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the input signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. Extensive empirical studies based on a synthetic dataset and a real power plant dataset demonstrate that MSCRED can outperform state-ofthe-art baseline methods.
    - 将原始序列转变为signature matrix. 通过inner product 的方式, 重构matrix而不是重构原始序列




----
<span id="cluster"> </span>
> ## **Clustering Analysis**  
<span id="deyubo"> </span>
- *Structural Deep Clustering Network* [^^^](#return)
    - https://dl.acm.org/doi/10.1145/3366423.3380214
    - Deyu Bo, Xiao Wang, Chuan SHi, Meiqi Zhu, Emiao Lu, Peng Cui
    - WWW '20: Proceedings of The Web Conference 2020, April 2020, Pages 1400–1410
    - Clustering is a fundamental task in data analysis. Recently, deep clustering, which derives inspiration primarily from deep learning approaches, achieves state-of-the-art performance and has attracted considerable attention. Current deep clustering methods usually boost the clustering results by means of the powerful representation ability of deep learning, e.g., autoencoder, suggesting that learning an effective representation for clustering is a crucial requirement. The strength of deep clustering methods is to extract the useful representations from the data itself, rather than the structure of data, which receives scarce attention in representation learning. Motivated by the great success of Graph Convolutional Network (GCN) in encoding the graph structure, we propose a Structural Deep Clustering Network (SDCN) to integrate the structural information into deep clustering. Specifically, we design a delivery operator to transfer the representations learned by autoencoder to the corresponding GCN layer, and a dual self-supervised mechanism to unify these two different deep neural architectures and guide the update of the whole model. In this way, the multiple structures of data, from low-order to high-order, are naturally combined with the multiple representations learned by autoencoder. Furthermore, we theoretically analyze the delivery operator, i.e., with the delivery operator, GCN improves the autoencoder-specific representation as a high-order graph regularization constraint and autoencoder helps alleviate the over-smoothing problem in GCN. Through comprehensive experiments, we demonstrate that our propose model can consistently perform better over the state-of-the-art techniques.
    - 基于自编码器和图网络进行特征表征学习，分两路，一路是用自编码器得到每层编码的特征H， 另一路是用GCN进行表征学习结构化特征Z，在GCN和AE的每层进行交互 ，参考公式（6），提出一种对偶自监督模块。 损失函数为类别的目标分布与网络计算分布的差异，用KL散度衡量，具体参考文章3.4节。Target distribution 通过计算得出。

<span id="junyuanxie"> </span>
- *Unsupervised deep embedding for clustering analysis* [^^^](#return)
    - https://dl.acm.org/doi/10.5555/3045390.3045442
    - Junyuan Xie, Ross Girshick, Ali Farhadi
    - ICML'16: Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48 June 2016 Pages 478–487
    - Clustering is central to many data-driven application domains and has been studied extensively in terms of distance functions and grouping algorithms. Relatively little work has focused on learning representations for clustering. In this paper, we propose Deep Embedded Clustering (DEC), a method that simultaneously learns feature representations and cluster assignments using deep neural networks. DEC learns a mapping from the data space to a lower-dimensional feature space in which it iteratively optimizes a clustering objective. Our experimental evaluations on image and text corpora show significant improvement over state-of-the-art methods.
    - 高提出深度嵌入聚类(Deep Embedded Clustering), 第一步度量嵌入表征与给定的聚类中心，第二步更新模型参数以及聚类中心基于额外的目标分布。核心在于如何定义target distribution P, 期望分布具有以下特性：强预测即聚类纯度要高、将更多的数据划分为高置信度：正则化损失贡献避免大的类别破坏隐特征空间。评价指标选择为ACC。在实验部分对多种因素进行了讨论。
 

---
<span id="featureis"> </span>
> ## **Feature Interaction&Selection**

---
<span id="cv"> </span>
> ## **Computer Vision (CV)** 
>> ### Image Classification/Detection/Applications/Networks
<span id="alex2012"> </span>
- *ImageNet Classification With Deep Convolutional Neural Networks* [^^^](#return)
    - https://dl.acm.org/doi/10.5555/2999134.2999257 
    - Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    - NIPS'12: Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1December 2012 Pages 1097–1105
    - We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overriding in the fully-connected layers we employed a recently-developed regularization method called "dropout" that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.
    - ImagNet, 一个标志性的卷积网络，开起了深度学习的时代，用于图像识别

<span id="colin2017"> </span>
- *Temporal Convolutional Networks for Action Segmentation and Detection* [^^^](#return)
    - https://ieeexplore.ieee.org/document/8099596 
    - Colin Lea; Michael D. Flynn; René Vidal; Austin Reiter; Gregory D. Hager
    - 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
    - The ability to identify and temporally segment fine-grained human actions throughout a video is crucial for robotics, surveillance, education, and beyond. Typical approaches decouple this problem by first extracting local spatiotemporal features from video frames and then feeding them into a temporal classifier that captures high-level temporal patterns. We describe a class of temporal models, which we call Temporal Convolutional Networks (TCNs), that use a hierarchy of temporal convolutions to perform fine-grained action segmentation or detection. Our Encoder-Decoder TCN uses pooling and upsampling to efficiently capture long-range temporal patterns whereas our Dilated TCN uses dilated convolutions. We show that TCNs are capable of capturing action compositions, segment durations, and long-range dependencies, and are over a magnitude faster to train than competing LSTM-based Recurrent Neural Networks. We apply these models to three challenging fine-grained datasets and show large improvements over the state of the art.
    - 提出时空卷积网络（TCN）

<span id="minlin2014"> </span>
- *Network in Network* [^^^](#return)
    - https://arxiv.org/abs/1312.4400 
    - Min Lin, Qiang Chen, Shuicheng Yan
    - arXiv, 2014
    - We propose a novel deep network structure called "Network In Network" (NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator. The feature maps are obtained by sliding the micro networks over the input in a similar manner as CNN; they are then fed into the next layer. Deep NIN can be implemented by stacking mutiple of the above described structure. With enhanced local modeling via the micro network, we are able to utilize global average pooling over feature maps in the classification layer, which is easier to interpret and less prone to overfitting than traditional fully connected layers. We demonstrated the state-of-the-art classification performances with NIN on CIFAR-10 and CIFAR-100, and reasonable performances on SVHN and MNIST datasets.
    - 1.用多层感知机来代替传统的采用线性内积计算的卷积核，用于更好的捕获局部的抽象特征。2.该micro network依然采用卷积核的方式在图像上滑动，且共享参数。3.提出Global Average Pooling (GAP) 减少网络参数，降低过拟合的风险，具有一定的可解释
 
 <span id="xiaolongwang2018"> </span>
- *Non-local Neural Networks* [^^^](#return)
    - https://arxiv.org/abs/1711.07971 
    - https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html 
    - https://github.com/facebookresearch/video-nonlocal-net 
    - Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He
    - Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7794-7803
    - Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our non-local models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks.

<span id="lecun1998"> </span>
- *Gradient-Based Learning Applied to Document Recognition*  [^^^](#return)
    - https://ieeexplore.ieee.org/document/726791 
    - Y. Lecun; L. Bottou; Y. Bengio; P. Haffner
    - in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
    - Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day.
    - 提出LeNet-5, 卷积网络成功的应用

<span id="matthew2014"> </span>
- *Visualizing and Understanding Convolutional Networks* [^^^](#return)
    - https://link.springer.com/chapter/10.1007%2F978-3-319-10590-1_53
    - https://arxiv.org/abs/1311.2901 
    - Matthew D. ZeilerRob Fergus
    - European Conference on Computer Vision (ECCV), pp 818-833, 2014. booktitle="Computer Vision -- ECCV 2014"
    - Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark Krizhevsky et al. However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we explore both issues. We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. Used in a diagnostic role, these visualizations allow us to find model architectures that outperform Krizhevsky et al on the ImageNet classification benchmark. We also perform an ablation study to discover the performance contribution from different model layers. We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.
    - 卷积网络的可视化工作

<span id="evan2017"> </span>
- *Fully Convolutional Networks for Semantic Segmentation* [^^^](#return)
    - https://ieeexplore.ieee.org/document/7478072 
    - Evan Shelhamer, Jonathan Long; Trevor Darrell
    - IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) ( Volume: 39, Issue: 4, 2017) 640 - 651
    - Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, improve on the previous best result in semantic segmentation. Our key insight is to build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional networks achieve improved segmentation of PASCAL VOC (30% relative improvement to 67.2% mean IU on 2012), NYUDv2, SIFT Flow, and PASCAL-Context, while inference takes one tenth of a second for a typical image.
    - 提出全卷积网络FCN

<span id="karen2015"> </span>
- *Very deep convolutional networks for large-scale image recognition* [^^^](#return)
    - https://arxiv.org/abs/1409.1556
    - Karen Simonyan, Andrew Zisserman
    - 3rd International Conference on Learning Representations, ICLR 2015
    - In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.
    - 研究网络深度对网络性能的影响，采用了较小的卷积核，提出VGG16, VGG19，研究表明网络的深度有利于分类精度


<span id="cv2"> </span>
>> ### Image Representation Learning ((Dis)entangled)
<span id="francesco2019"> </span>
- *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* [^^^](#return)
    - https://arxiv.org/abs/1811.12359 
    - https://github.com/google-research/disentanglement_lib 
    - Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem
    - Proceedings of the 36th International Conference on Machine Learning (ICML 2019), 7247-7283
    - The key idea behind the unsupervised learning of disentangled representations is that real-world data is generated by a few explanatory factors of variation which can be recovered by unsupervised learning algorithms. In this paper, we provide a sober look at recent progress in the field and challenge some common assumptions. We first theoretically show that the unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both the models and the data. Then, we train more than 12000 models covering most prominent methods and evaluation metrics in a reproducible large-scale experimental study on seven different data sets. We observe that while the different methods successfully enforce properties ``encouraged'' by the corresponding losses, well-disentangled models seemingly cannot be identified without supervision. Furthermore, increased disentanglement does not seem to lead to a decreased sample complexity of learning for downstream tasks. Our results suggest that future work on disentanglement learning should be explicit about the role of inductive biases and (implicit) supervision, investigate concrete benefits of enforcing disentanglement of the learned representations, and consider a reproducible experimental setup covering several data sets.
    - 讨论解耦表征学习中的一些问题，表明 (没有归纳偏置的) 无监督方法学不到可靠的分离式表征，本篇是ICML2019的两篇best paper之一。

<span id="hyunjik2018"> </span>
- *Disentangling by Factorising* [^^^](#return)
    - http://proceedings.mlr.press/v80/kim18b.html 
    - Hyunjik Kim, Andriy Mnih
    - Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2649-2658, 2018.
    - We define and address the problem of unsupervised learning of disentangled representations on data generated from independent factors of variation. We propose FactorVAE, a method that disentangles by encouraging the distribution of representations to be factorial and hence independent across the dimensions. We show that it improves upon beta-VAE by providing a better trade-off between disentanglement and reconstruction quality and being more robust to the number of training iterations. Moreover, we highlight the problems of a commonly used disentanglement metric and introduce a new metric that does not suffer from them.
    - 鼓励学习各个表征的分布是因子的（factorial），从而实现表征的解耦和独立

<span id="xichen2016"> </span>
- *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets* [^^^](#return)
    - https://dl.acm.org/doi/10.5555/3157096.3157340 
    - https://arxiv.org/abs/1606.03657 
    - Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    - NIPS'16: Proceedings of the 30th International Conference on Neural Information Processing Systems December 2016 Pages 2180-2188
    - This paper describes InfoGAN, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation. We derive a lower bound to the mutual information objective that can be optimized efficiently, and show that our training procedure can be interpreted as a variation of the Wake-Sleep algorithm. Specifically, InfoGAN successfully disentangles writing styles from digit shapes on the MNIST dataset, pose from lighting of 3D rendered images, and background digits from the central digit on the SVHN dataset. It also discovers visual concepts that include hair styles, presence/absence of eyeglasses, and emotions on the CelebA face dataset. Experiments show that InfoGAN learns interpretable representations that are competitive with representations learned by existing fully supervised methods.
    - 在GAN的基础上进行改进，由于GAN中的z是一个连续的噪声信号，没有任何约束，所以GAN无法利用这个z来进行生成特定数据，InfoGAN希望做到隐变量是具有可解释性的一个表达，于是将输入的噪声进行分解，分解为 不可压缩的噪声z和可解释的隐变量c，希望可以通过约束c和生成数据之间的关系，使得c具有对数据的可解释信息。

<span id="ricky2018"> </span>
- *Isolating Sources of Disentanglement in VAEs* [^^^](#return)
    - https://arxiv.org/abs/1802.04942 
    - https://blog.csdn.net/qq_31239495/article/details/82702303 
    - Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud
    - arXiv, 2018
    - We decompose the evidence lower bound to show the existence of a term measuring the total correlation between  latent variables. We use this to motivate our β-TCVAE (Total Correlation Variational Autoencoder), a refinement of the state-of-the-art β-VAE objective for learning disentangled representations, requiring no additional hyperparameters during training. We further propose a principled classifier-free measure of disentanglement called the mutual information gap (MIG). We perform extensive quantitative and qualitative experiments, in both restricted and non-restricted settings, and show a strong relation between total correlation and disentanglement, when the latent variables model is trained using our framework.
    - 提出β-TCVAE，将 ELBO(evidence lower bound)分解成多项，用于调整隐变量之间的关系，提出 β-TCVAE 算法，是 β-VAE 的加强和替换版本，并且在训练中不增加任何超参数。论文进一步提出 disentanglement 的规则的无分类方法 MIG( mutaul information gap)。论文主要做了四个贡献：① 分解 ELBO， 解释 β-VAE 的成功之处 ② 提出一个方法：基于随机训练中的权重采样，且不增加任何超参数 ③ 引入 β-TCVAE 发现更多可解释隐变量，在随机初始化情况下具有更强的鲁棒性 ④ 从信息论视角处理 disentanglement ，无分类器和可生成随机分布和无标准分布的隐变量。

<span id="aaronvanden2017"> </span>
- *Neural Discrete Representation Learning* [^^^](#return)
    - https://arxiv.org/abs/1711.00937 
    - https://papers.nips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html 
    - Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
    - Advances in Neural Information Processing Systems 2017, 30, 6306--6315
    - Learning useful representations without supervision remains a key challenge in machine learning. In this paper, we propose a simple yet powerful generative model that learns such discrete representations. Our model, the Vector Quantised- Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. In order to learn a discrete latent representation, we incorporate ideas from vector quantisation (VQ). Using the VQ method allows the model to circumvent issues of “posterior collapse” -— where the latents are ignored when they are paired with a powerful autoregressive decoder -— typically observed in the VAE framework. Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations.
    - 提出VQ-VAE, 学习的是离散表征，而非连续表征

<span id="irina2017"> </span>
- *β-VAE: Learning Basic Visual Concepts With A Constrained Variational Framework* [^^^](#return)
    - https://openreview.net/forum?id=Sy2fzU9gl 
    - Irina Higginsirinah，Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner
    - ICLR2017
    - Learning an interpretable factorised representation of the independent data generative factors of the world without supervision is an important precursor for the development of artificial intelligence that is able to learn and reason in the same way that humans do. We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner. Our approach is a modification of the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter beta that balances latent channel capacity and independence constraints with reconstruction accuracy. We demonstrate that beta-VAE with appropriately tuned beta > 1 qualitatively outperforms VAE (beta = 1), as well as state of the art unsupervised (InfoGAN) and semi-supervised (DC-IGN) approaches to disentangled factor learning on a variety of datasets (celebA, faces and chairs). Furthermore, we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models, and show that our approach also significantly outperforms all baselines quantitatively. Unlike InfoGAN, beta-VAE is stable to train, makes few assumptions about the data and relies on tuning a single hyperparameter, which can be directly optimised through a hyper parameter search using weakly labelled data or through heuristic visual inspection for purely unsupervised data.
    - 提出β-VAE，用于表征学习

<span id="christopher2018"> </span>
- *Understanding disentangling in β-VAE* [^^^](#return)
    - https://arxiv.org/abs/1804.03599
    - Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, Alexander Lerchner 
    - arXiv:1804.03599, 2018. Presented at the 2017 NIPS Workshop on Learning Disentangled Representations
    -We present new intuitions and theoretical assessments of the emergence of disentangled representation in variational autoencoders. Taking a rate-distortion theory perspective, we show the circumstances under which representations aligned with the underlying generative factors of variation of data emerge when optimising the modified ELBO bound in β-VAE, as training progresses. From these insights, we propose a modification to the training regime of β-VAE, that progressively increases the information capacity of the latent code during training. This modification facilitates the robust learning of disentangled representations in β-VAE, without the previous trade-off in reconstruction accuracy.
    - 对β-VAE的解释说明


---
<span id="nlp"> </span>
> ## **Natural Languages Processing (NLP/Sequence Model)** 
>> ### Languages Classification/Applications/Networks
<span id="sepp1997"> </span>
- *Long Short-Term Memory* [^^^](#return)
    - https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735 
    - Sepp Hochreiter, Jürgen Schmidhuber
    - Neural Computation, 1997, 9(8), 1735-1780
    - Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O. 1. Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.
    - 提出LSTM

<span id="shiyuchang2017"> </span>
- *Dilated Recurrent Neural Networks* [^^^](#return)
    - https://arxiv.org/abs/1710.02224 
    - https://github.com/code-terminator/DilatedRNN 
    - https://papers.nips.cc/paper/2017/hash/32bb90e8976aab5298d5da10fe66f21d-Abstract.html 
    - Shiyu Chang, Yang Zhang, Wei Han, Mo Yu, Xiaoxiao Guo, Wei Tan, Xiaodong Cui, Michael Witbrock, Mark Hasegawa-Johnson, Thomas S. Huang
    - Advances in Neural Information Processing Systems 30 (NIPS 2017)
    - Learning with recurrent neural networks (RNNs) on long sequences is a notoriously difficult task. There are three major challenges: 1) complex dependencies, 2) vanishing and exploding gradients, and 3) efficient parallelization. In this paper, we introduce a simple yet effective RNN connection structure, the DilatedRNN, which simultaneously tackles all of these challenges. The proposed architecture is characterized by multi-resolution dilated recurrent skip connections and can be combined flexibly with diverse RNN cells. Moreover, the DilatedRNN reduces the number of parameters needed and enhances training efficiency significantly, while matching state-of-the-art performance (even with standard RNN cells) in tasks involving very long-term dependencies. To provide a theory-based quantification of the architecture's advantages, we introduce a memory capacity measure, the mean recurrent length, which is more suitable for RNNs with long skip connections than existing measures. We rigorously prove the advantages of the DilatedRNN over other recurrent neural architectures.
    - 提出扩展RNN

<span id="shaojiebai"> </span>
- *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* [^^^](#return)
    - https://arxiv.org/abs/1803.01271v1 
    - https://github.com/locuslab/TCN 
    - Shaojie Bai, J. Zico Kolter, Vladlen Koltun
    - arXiv, 2018.
    - For most deep learning practitioners, sequence modeling is synonymous with recurrent networks. Yet recent results indicate that convolutional architectures can outperform recurrent networks on tasks such as audio synthesis and machine translation. Given a new sequence modeling task or dataset, which architecture should one use? We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks.
    - 对于大多数深度学习从业人员而言，序列建模是循环网络的同义词。然而，最近的结果表明，在诸如音频合成和机器翻译之类的任务上，卷积架构的性能要优于循环网络。 给定一个新的序列建模任务或数据集，应该使用哪种架构？ 我们对通用卷积和循环体系进行序列建模的系统评估。 对模型进行评估的范围广泛，这些任务通常用于对循环网络进行基准测试。 我们的结果表明，简单的卷积体系结构在各种任务和数据集上均表现出优于LSTMs的规范化循环网络，同时展示了更长的有效内存。 我们得出的结论是，应该重新考虑序列建模与循环网络之间的常见关联，并且应该将卷积网络视为序列建模任务的自然起点。


>> ### Words Representation Learning


---
<span id="rs"> </span>
> ## **Recommender Systems**


---
<span id="ml"> </span>
> ## **Machine Learning**
<span id="guolinke"> </span>
- *LightGBM: A highly efficient gradient boosting decision tree* [^^^](#return)
    - https://dl.acm.org/doi/10.5555/3294996.3295074
    - https://lightgbm.readthedocs.io/en/latest/ 
    - Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu
    - NIPS'17: Proceedings of the 31st International Conference on Neural Information Processing Systems, 3149–3157
    - Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). We call our new GBDT implementation with GOSS and EFB LightGBM. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.
    - 提出LightGBM, 一种工业实现版本的GBDT

- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* [^^^](#return)
    - https://jmlr.csail.mit.edu/papers/v15/srivastava14a.html 
    - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
    - Journal of Machine Learning Research, 15(56):1929−1958, 2014.
    - Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different âthinnedâ networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets.
    - 提出Dropout来避免神经网络的过拟合

- *XGBoost: A Scalable Tree Boosting System*
    - Qi

---
<span id="fdd"> </span>
> ## **Fault Diagnosis and Detection**
<span id="zhongxuhu2020"> </span>
- *Data-Driven Fault Diagnosis Method Based on Compressed Sensing and Improved Multiscale Network* [^^^](#return)
    - https://ieeexplore.ieee.org/document/8704327
    - Zhong-Xu Hu, Yan Wang, Ming-Feng Ge, Jie Liu
    - IEEE Transactions on Industrial Electronics ( Volume: 67, Issue: 4, April 2020), 3216 – 3225
    - The diagnosis of the key components of rotating machinery systems is essential for the production efficiency and quality of manufacturing processes. The performance of the traditional diagnosis method depends heavily on feature extraction, which relies on the degree of individual's expertise or prior knowledge. Recently, a deep learning (DL) method is applied to automate feature extraction. However, training in the DL method requires a massive amount of sensor data, which is time consuming and poses a challenge for its applications in engineering. In this paper, a new data-driven fault diagnosis method based on compressed sensing (CS) and improved multiscale network (IMSN) is proposed to recognize and classify the faults in rotating machinery. CS is used to reduce the amount of raw data, from which the fault information is discovered. At the same time, it can be used to generate sufficient training samples for the subsequent learning. The one-dimensional compressed signal is converted to two-dimensional image for further learning. An IMSN is established for learning and obtaining deep features. It improves the diagnosis performance of the DL process. The faults of the key components are identified from a softmax model. Experimental analysis is performed to verify effectiveness of the proposed data-driven fault diagnosis method.
    - 提出一种基于压缩感知和改进多尺度网络的数据故障诊断方法。基于the measurement matrix of the CS，也就是压缩感知的测量矩阵，将compressed sensing用于数据的信息抽取和训练数据的生成，将一维的信号转变为二维的图像数据，IMSN用于学习深度特征提取，网络中包含了特征图的拼接。
    - 2-D image is constructed by randomly choosing data segments from the compressed vibration signal
    - 本质上是对原始数据进行压缩，乘以一个压缩矩阵，实际上是个映射，Multiplying the original signal by the observation matrix increases the spacing between data
    - 针对单变量，缺乏temporal information，且需要label

<span id="haowu"> </span>
- *Deep Convolutional Neural Network Model based Chemical Process Fault Diagnosis* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S0098135418302990?via%3Dihub
    - Hao Wu, Jinsong Zhao
    - Computers & Chemical Engineering, Volume 115, 12 July 2018, Pages 185-197
    - Numerous accidents in chemical processes have caused emergency shutdowns, property losses, casualties and/or environmental disruptions in the chemical process industry. Fault detection and diagnosis (FDD) can help operators timely detect and diagnose abnormal situations, and take right actions to avoid ad- verse consequences. However, FDD is still far from widely practical applications. Over the past few years, deep convolutional neural network (DCNN) has shown excellent performance on machine-learning tasks. In this paper, a fault diagnosis method based on a DCNN model consisting of convolutional layers, pooling layers, dropout, fully connected layers is proposed for chemical process fault diagnosis. The benchmark Tennessee Eastman (TE) process is utilized to verify the outstanding performance of the fault diagnosis method.
    - 卷积网络用于TE Process，主要是看如何仿真TE Process, t-SNE 可用来可视化

<span id= ""> </span>
- *A deep belief network based fault diagnosis model for complex chemical processes* [^^^](#return)
    - https://www.sciencedirect.com/science/article/pii/S0098135417301059
    - Zhanpeng Zhang, Jinsong Zhao
    - Computers & Chemical Engineering, Volume 107, 5 December 2017, Pages 395-407
    - Data-driven methods have been regarded as desirable methods for fault detection and diagnosis (FDD) of practical chemical processes. However, with the big data era coming, how to effectively extract and present fault features is one of the keys to successful industrial applications of FDD technologies. In this paper, an extensible deep belief network (DBN) based fault diagnosis model is proposed. Individual fault features in both spatial and temporal domains are extracted by DBN sub-networks, aided by the mutual information technology. A global two-layer back-propagation network is trained and used for fault classification. In the final part of this paper, the benchmarked Tennessee Eastman process is utilized to illustrate the performance of the DBN based fault diagnosis model.
    - 深度置信网络用于提取故障特征，参考如何仿真TE Process


---
<span id="valve"> </span>
> ## **Valve Stiction**
>> ### Detection and Quantification
- *A Curve Fitting Method for Detecting Valve Stiction in Oscillating Control Loops* [^^^](#return)
     - https://pubs.acs.org/doi/10.1021/ie061219a 
     - Q. Peter He, Jin Wang, Martin Pottmann, S. Joe Qin
     - Industrial & Engineering Chemistry Research. Ind. Eng. Chem. Res. 2007, 46, 13, 4549–4560
     - Many control loops in process plants perform poorly because of valve stiction as one of the most common equipment problems. Valve stiction may cause oscillation in control loops, which increases variability in product quality, accelerates equipment wear, or leads to control system instability and other issues that potentially disrupt the operation. In this work, data-driven valve stiction models are first reviewed and a simplified model is presented. Next, a stiction detection method is proposed based on curve fitting of the output signal of the first integrating component after the valve, i.e., the controller output for self-regulating processes or the process output for integrating processes. A metric that is called the stiction index (SI) is introduced, based on the proposed method to facilitate the automatic detection of valve stiction. The effectiveness of the proposed method is demonstrated using both simulated data sets based on the proposed valve stiction model and real industrial data sets.
    - 


>> ### Compensation and Control



