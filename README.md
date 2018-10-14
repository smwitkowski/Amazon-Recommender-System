
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Motivation" data-toc-modified-id="Motivation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Motivation</a></span></li><li><span><a href="#Tools-and-Libraries-Used" data-toc-modified-id="Tools-and-Libraries-Used-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Tools and Libraries Used</a></span></li><li><span><a href="#Data-Sources" data-toc-modified-id="Data-Sources-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Sources</a></span></li><li><span><a href="#Analysis" data-toc-modified-id="Analysis-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Analysis</a></span></li><li><span><a href="#Limitations" data-toc-modified-id="Limitations-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Limitations</a></span></li></ul></div>

# Collaborative Filtering Using Amazon Review Data

## Introduction
In this project, I use the `Surprise` package to create a recommender system using reviews of products in the "Watch" category on Amazon. This project and all the contents were developed and published for educational purposes only.

## Motivation
Many businesses can benefit vastly from implementing some sort of recommendation engine. I wanted to complete this project to show the relatively low amount of effort can go into such a fruitful internal product.

## Tools and Libraries Used
While recommender systems can be built in many different languages, I elected to use Python to build this one. I use a few different popular tools and libraries to complete this project:

* Jupyter Notebook 5.3.1
* pandas 0.21.0
* Python 3.6.2
* scikit-surprise 1.0.6

## Data Sources
I'm using a [public dataset of Amazon reviews](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). There's anonymized customer information, but the reviews and products are real. There data sets provided by Amazon are split up by product category. I chose to use reviews of products in the "Watch" category. Since the data is so large, I didn't include it in this GitHub repo, however it is [available online](https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

## Analysis
You can find my complete analysis in my [GitHub repo](Collaborative Filtering Amazon Watch Reviews). The file it titles [Collaborative Filtering Amazon Watch Reviews](https://github.com/smwitkowski/Amazon-Recommender-System/blob/master/Collaborative%20Filtering%20Amazon%20Watch%20Reviews.ipynb)

## Limitations
There were multiple hurdles I faced in this project. I detail them in my analysis, but I'll list them briefly here:

* I used my personal laptop, so optimizing parameters took a lot of time. ALso, I could not run a k-NN on this data because it was so large.
* The data doesn't have any product descriptions, so I could not do any sort of content filtering
* Many products have only been reviewed once, and many users have only reviewed one product, which leads to many sparse recommendations
