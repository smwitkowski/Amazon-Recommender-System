
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Loading Data</a></span></li><li><span><a href="#Removing-Duplicates" data-toc-modified-id="Removing-Duplicates-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Removing Duplicates</a></span></li><li><span><a href="#Checking-Total-Reviews" data-toc-modified-id="Checking-Total-Reviews-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Checking Total Reviews</a></span></li></ul></li><li><span><a href="#Model-Selection" data-toc-modified-id="Model-Selection-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Model Selection</a></span><ul class="toc-item"><li><span><a href="#Baseline-Estimate" data-toc-modified-id="Baseline-Estimate-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Baseline Estimate</a></span></li><li><span><a href="#Single-Value-Decomposition" data-toc-modified-id="Single-Value-Decomposition-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Single Value Decomposition</a></span></li></ul></li><li><span><a href="#Serving-Recommendations" data-toc-modified-id="Serving-Recommendations-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Serving Recommendations</a></span><ul class="toc-item"><li><span><a href="#Finding-Popular-Classes" data-toc-modified-id="Finding-Popular-Classes-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Finding Popular Classes</a></span></li></ul></li><li><span><a href="#Wrapping-Up" data-toc-modified-id="Wrapping-Up-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Wrapping Up</a></span></li></ul></div>

# Building a Recommender System Using Amazon Reviews 

## Introduction
Recommendation systems are in tons of things you interact with every day. Amazon, Spotify, and Facebook are some of the biggest players, and they're using all the data they can to suggest products that they think you'll love.

<img src="images/spotify_recommendations.png">

Some companies have teams of people collection, cleaning, and building models around this data. However, with a few useful Python packages and some great data from [Amazon's customer review dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), I'm going to build a recommendation system by myself.

## Data Preprocessing

### Loading Data
There is **a lot** of data in Amazon's full customer review dataset, so I'm only going to be using data on the 'Watches' category.


```python
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Read in the data, skip any lines that return an error
reviews = pd.read_csv(
    'data\Watch Reviews.tsv',
    sep='\t',
    error_bad_lines=False,
    warn_bad_lines=False)
reviews.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>marketplace</th>
      <th>customer_id</th>
      <th>review_id</th>
      <th>product_id</th>
      <th>product_parent</th>
      <th>product_title</th>
      <th>product_category</th>
      <th>star_rating</th>
      <th>helpful_votes</th>
      <th>total_votes</th>
      <th>vine</th>
      <th>verified_purchase</th>
      <th>review_headline</th>
      <th>review_body</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>3653882</td>
      <td>R3O9SGZBVQBV76</td>
      <td>B00FALQ1ZC</td>
      <td>937001370</td>
      <td>Invicta Women's 15150 "Angel" 18k Yellow Gold ...</td>
      <td>Watches</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
      <td>Y</td>
      <td>Five Stars</td>
      <td>Absolutely love this watch! Get compliments al...</td>
      <td>2015-08-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>14661224</td>
      <td>RKH8BNC3L5DLF</td>
      <td>B00D3RGO20</td>
      <td>484010722</td>
      <td>Kenneth Cole New York Women's KC4944 Automatic...</td>
      <td>Watches</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
      <td>Y</td>
      <td>I love thiswatch it keeps time wonderfully</td>
      <td>I love this watch it keeps time wonderfully.</td>
      <td>2015-08-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>27324930</td>
      <td>R2HLE8WKZSU3NL</td>
      <td>B00DKYC7TK</td>
      <td>361166390</td>
      <td>Ritche 22mm Black Stainless Steel Bracelet Wat...</td>
      <td>Watches</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>N</td>
      <td>Y</td>
      <td>Two Stars</td>
      <td>Scratches</td>
      <td>2015-08-31</td>
    </tr>
  </tbody>
</table>
</div>



Theres are 15 columns in the data. Amazon details what each columns containes, which I've included below:

* marketplace       - 2 letter country code of the marketplace where the review was written.
* customer_id       - Random identifier that can be used to aggregate reviews written by a single author.
* review_id         - The unique ID of the review.
* product_id        - The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
* product_parent    - Random identifier that can be used to aggregate reviews for the same product.
* product_title     - Title of the product.
* product_category  - Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
* star_rating       - The 1-5 star rating of the review.
* helpful_votes     - Number of helpful votes.
* total_votes       - Number of total votes the review received.
* vine              - Review was written as part of the Vine program.
* verified_purchase - The review is on a verified purchase.
* review_headline   - The title of the review.
* review_body       - The review text.
* review_date       - The date the review was written.


```python
reviews.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 960204 entries, 0 to 960203
    Data columns (total 15 columns):
    marketplace          960204 non-null object
    customer_id          960204 non-null int64
    review_id            960204 non-null object
    product_id           960204 non-null object
    product_parent       960204 non-null int64
    product_title        960202 non-null object
    product_category     960204 non-null object
    star_rating          960204 non-null int64
    helpful_votes        960204 non-null int64
    total_votes          960204 non-null int64
    vine                 960204 non-null object
    verified_purchase    960204 non-null object
    review_headline      960197 non-null object
    review_body          960056 non-null object
    review_date          960200 non-null object
    dtypes: int64(5), object(10)
    memory usage: 109.9+ MB
    

### Removing Duplicates
All-in-all, this data is pretty clean! There are some records with missing data, but it won't cause any issues in our analysis.

I'm not certain how this data was collected or if it processed before it was published; there could be some duplicated reviews in here. To check, I'll see if there are any review_id duplicated in the data.


```python
sum(reviews.review_id.duplicated())
```




    0



There aren't any duplicated `review_id` values, but that doesn't mean that there aren't any duplicative reviews. Hypothetically, someone could submit the same review twice, and we wouldn't know only by looking at `review_id`.

Instead of looking at `review_id`, I'm going to see if a customer posted two reviews on the same product. My assumption is that is `customer_id` and `product_id` show up more than once, then it's a duplicated review.


```python
purchase_ids = ['customer_id', 'product_id']

# Get a dataframe consisting only of reviews that are duplicated
duplicates = reviews[reviews.duplicated(subset=purchase_ids,
                                        keep=False)].sort_values(purchase_ids)
duplicates.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>marketplace</th>
      <th>customer_id</th>
      <th>review_id</th>
      <th>product_id</th>
      <th>product_parent</th>
      <th>product_title</th>
      <th>product_category</th>
      <th>star_rating</th>
      <th>helpful_votes</th>
      <th>total_votes</th>
      <th>vine</th>
      <th>verified_purchase</th>
      <th>review_headline</th>
      <th>review_body</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>180410</th>
      <td>US</td>
      <td>685318</td>
      <td>R1J1TVEYBP3A7M</td>
      <td>B003QG1SO2</td>
      <td>47470030</td>
      <td>Timex Men's Easy Reader Date Leather Strap Watch</td>
      <td>Watches</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>N</td>
      <td>Y</td>
      <td>Three Stars</td>
      <td>it's ok, but there's one part below the the nu...</td>
      <td>2015-03-10</td>
    </tr>
    <tr>
      <th>227516</th>
      <td>US</td>
      <td>685318</td>
      <td>R1DJUOOB0RFMRP</td>
      <td>B003QG1SO2</td>
      <td>47470030</td>
      <td>Timex Men's Easy Reader Date Leather Strap Watch</td>
      <td>Watches</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
      <td>Y</td>
      <td>Four Stars</td>
      <td>it's nice.</td>
      <td>2015-02-01</td>
    </tr>
    <tr>
      <th>45889</th>
      <td>US</td>
      <td>817344</td>
      <td>R3R845UJQPL7DP</td>
      <td>B004VW55NA</td>
      <td>921527551</td>
      <td>Women Geneva Rhinestone Leather Band Quartz Wr...</td>
      <td>Watches</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
      <td>Y</td>
      <td>Three Stars</td>
      <td>nice &amp; shiny</td>
      <td>2015-07-20</td>
    </tr>
    <tr>
      <th>46029</th>
      <td>US</td>
      <td>817344</td>
      <td>R39DJJ1LM1K14N</td>
      <td>B004VW55NA</td>
      <td>921527551</td>
      <td>Women Geneva Rhinestone Leather Band Quartz Wr...</td>
      <td>Watches</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
      <td>Y</td>
      <td>Three Stars</td>
      <td>Nice</td>
      <td>2015-07-20</td>
    </tr>
  </tbody>
</table>
</div>



So it looks like there are some reviews that are exactly the same, but some people have *updated* their review by submitting a new one. With that in mind, I'm only going to keep the most recent review.


```python
reviews = (reviews
           # Sort the values so we'll keep the most recent review.
           .sort_values(['customer_id', 'product_id', 'review_date'], ascending=[False, False, True])
           .drop_duplicates(subset=purchase_ids, keep='last'))
```

Next, I want to look at the products that have been reviewed in this data. I expect that some products have been reviewed many times, while others might have only been reviewed once.


```python
reviews.product_title.value_counts().to_frame().head(5)
reviews.product_title.value_counts().to_frame().tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SE JT6216 21-Piece Watch Repair Kit</th>
      <td>4390</td>
    </tr>
    <tr>
      <th>Timex Unisex Weekender Analog Quartz Watch</th>
      <td>3229</td>
    </tr>
    <tr>
      <th>Bling Jewelry Plated Classic Round CZ Ladies Watch</th>
      <td>3050</td>
    </tr>
    <tr>
      <th>Casio Men's Sport Analog Dive Watch</th>
      <td>2047</td>
    </tr>
    <tr>
      <th>Casio Women's LRW-200H-2BVCF Stainless Steel Watch Resin Band</th>
      <td>2008</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Burberry Men's BU7721 Sport Diving Blue Diving Dial Watch</th>
      <td>1</td>
    </tr>
    <tr>
      <th>TAG Heuer Women's WJ1316.BA0573 Link Watch</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Akribos XXIV Bravura Mens Watch AK480SS</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Ebel Women's Quartz Watch 9157F13-9925</th>
      <td>1</td>
    </tr>
    <tr>
      <th>K&amp;BROS Women's 9149-1 Steel Flower Stainless Steel Black Dial Watch</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Interestingly, the top reviewed item is a repair kit! While this isn't a watch, I'm not going to remove it because it's in the "Watch" category on Amazon.

Now, something worth noting is that each product comes with a `parent_product` value. I expect that this is main product in an Amazon listing, and child products are those of different sizes or different colors. I'm going to do some digging and see if that's the case.


```python
reviews[['product_parent',
         'product_id']].drop_duplicates().product_parent.value_counts().head(5)
```




    297568235    48
    315669547    48
    324241921    37
    802598355    35
    995353619    31
    Name: product_parent, dtype: int64



Theres a lot of different products that share `product_parent`. I'm going to look at the titles of one of them.


```python
reviews[reviews.product_parent == 297568235][[
    'product_parent', 'product_id', 'product_title'
]].drop_duplicates().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_parent</th>
      <th>product_id</th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47769</th>
      <td>297568235</td>
      <td>B005DM0PHS</td>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
    <tr>
      <th>644527</th>
      <td>297568235</td>
      <td>B005BDZFBA</td>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
    <tr>
      <th>690385</th>
      <td>297568235</td>
      <td>B005BKNGCI</td>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
    <tr>
      <th>849420</th>
      <td>297568235</td>
      <td>B005CHHTE6</td>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
    <tr>
      <th>849105</th>
      <td>297568235</td>
      <td>B005CHHKUE</td>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
  </tbody>
</table>
</div>



There's a lot of products for some Clockwork Synergy product. I wonder if they're different products or not.


```python
reviews[reviews.product_parent == 297568235][['product_title'
                                              ]].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47769</th>
      <td>Clockwork Synergy Classic Nylon Nato watch str...</td>
    </tr>
  </tbody>
</table>
</div>



So it seems that `product_parent` distinguishes between different products since there are different values for `product_id`. However, all of these products share the same value for `product_title`. It seems that the different `product_id` values represent differences in things colors or sizes, like in the product example below.

<img src="images/nylon_selections.png">

This could cause a problem in the future; a user theoretically could be recommended three products with unique `product_id` values, but they're really all the same product. I want to recommend three different products instead.

The main issue here is that there are many different values of `product_id` for each value of `product_title`. I don't want to remove any products or reviews, so instead I'm going to alter the data a bit to make sure that there is only one `product_id` per `product_title`.


```python
products = reviews[['product_id', 'product_title']].drop_duplicates(
    subset='product_title', keep='first')
column_order = reviews.columns
reviews = reviews.drop(
    'product_id', axis=1).merge(
        products, on='product_title')[column_order]

reviews[['product_id', 'product_title'
         ]].drop_duplicates().product_title.value_counts().head(5).to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ice-Watch Ice-Glow Glow Yellow - Big Men's watch #GL.GY.B.S.11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Victorinox Swiss Army Men's 241261 Classic Chronograph Black Dial Watch</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Nautica Unisex N09920G BFD 101 Classic Analog with Enamel Bezel Watch</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Swiss Legend Men's 30465-01-RDA "Cyclone" Stainless Steel Watch with Black Silicone Strap</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Marc Ecko Men's E95042G2 Rhino Logo Silver Stainless Steel Watch</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now we only have one `product_id` per `product_title`. I'm going to check and see if there are still multiple `product_id` values for one `product_parent` value.


```python
reviews[['product_parent',
         'product_id']].drop_duplicates().product_parent.value_counts().head(5)
```




    544812143    23
    193303569     9
    86574288      9
    286396053     7
    459113922     7
    Name: product_parent, dtype: int64



It seem so, but since we changed the `product_id` value for items with matching `product_title` values, I expect these are different products.


```python
reviews[reviews.product_parent == 544812143][['product_title'
                                              ]].drop_duplicates().head()

reviews[reviews.product_parent == 286396053][['product_title'
                                              ]].drop_duplicates().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>353378</th>
      <td>Game Time Men's NFL Retro Series Watch - India...</td>
    </tr>
    <tr>
      <th>569449</th>
      <td>Game Time Men's NFL Retro Series Watch</td>
    </tr>
    <tr>
      <th>581501</th>
      <td>Game Time Men's NFL Retro Series Watch - Phila...</td>
    </tr>
    <tr>
      <th>697318</th>
      <td>Game Time Men's NFL Retro Series Watch - New E...</td>
    </tr>
    <tr>
      <th>712317</th>
      <td>Game Time Men's NFL Retro Series Watch - Houst...</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>312536</th>
      <td>Seiko Men's SKZ211K1 Five Sports Stainless Ste...</td>
    </tr>
    <tr>
      <th>861913</th>
      <td>Seiko 5 Sport Black Dial Mens Watch SKZ211K1</td>
    </tr>
    <tr>
      <th>948609</th>
      <td>Seiko 5 Wrist Watches-Seiko 5 Sports Automatic...</td>
    </tr>
    <tr>
      <th>953379</th>
      <td>SKZ211 Seiko 5 Sports Automatic Atlas Diver Ye...</td>
    </tr>
    <tr>
      <th>955967</th>
      <td>Seiko 5 Men's Sports Automatic 200M Watch</td>
    </tr>
  </tbody>
</table>
</div>



So it looks like there are some products that are still very similar and share a `parent_product` value. The first set of parent id's showed different watches that highlighted different NFL teams.

<table><tr><td><img src='images/Arizona Retro.PNG'></td><td><img src='images/San Diego Retro.PNG'></td></tr></table>

These *could* be considered as the same product, but if you talk to a fan, an Arizona Caridinals watch **is not** the same as a New England Patriots watch.

The second set of `parent_product` values shows very different products.

<table><tr><td><img src='images/Seiko 1.PNG'></td><td><img src='images/Seiko 2.PNG'></td></tr></table>

I don't think it would ever make sense to group these watches together, so I'm going to keep the data as is.

### Checking Total Reviews
Many recommender systems run into a problem called the Cold-Start problem. Essentially, a user can't be recommended anything because they haven't rated anything! Additionally, if you introduce a new product, nobody has rated it and it can't be recommended.  Since we have rating data, we're not going to run into either of these problems, although we could face a similar one. 

If a user rated *one* item, how well do you think a recommender system could work? If you have one point of a line, you have no idea which direction the line is going. In the same vein, if you have a user with only one review, you *might* be able to rule out some items, but it would be very difficult to be confident in your recommendations to that user.

For this reason, I'm going to explore how many reviews have been submitted by user.


```python
(reviews.customer_id.value_counts().rename_axis('id').reset_index(
    name='frequency').frequency.value_counts(
        normalize=False).rename_axis('reviews').to_frame().head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>reviews</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>593745</td>
    </tr>
    <tr>
      <th>2</th>
      <td>81301</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22899</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9093</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2479</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1520</td>
    </tr>
    <tr>
      <th>8</th>
      <td>986</td>
    </tr>
    <tr>
      <th>9</th>
      <td>712</td>
    </tr>
    <tr>
      <th>10</th>
      <td>482</td>
    </tr>
  </tbody>
</table>
</div>



We can see that over 80% of our review dataset contains users that have only reviewed a single product. While we would like a lot of review data for every individual, data is data and we have to work with what we have. The good news is that over 80,000 people have reviewed at least 2 products, and we should be able to build a decent foundational model with the data we have in front of us.

## Model Selection

There are a lot of different packages that can help you build a recommender system. For this one, I'm going to be using the   [Surprise package](http://surpriselib.com/). `Surprise` has a few different algorithms built in. The author has even included some datasets, so you could get started building something similar without collecting or cleaning the data yourself. 

In this case, I need to load in a custom dataset to use with `Surprise`. It's not as easy as pointing to the dataframe, but it's pretty close. According to the [documentation](https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset), we need to make sure our dataframe has three columns: the user ids, the item ids, and the ratings. Additionally, we'll need to specify the rating scale. In our case, users can rate a product discretely from 1 to 5. 

I'm also going to split the data into training and testing data using the `Surprise` package


```python
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

# Set the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data from the review data frame and create train and test sets
data = Dataset.load_from_df(
    reviews[['customer_id', 'product_id', 'star_rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)
```

### Baseline Estimate

First, I'm going to build a recommendation system using baseline estimates. If you'd like to learn more about baseline estimates, you can read the [Surprise documentation](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) or [Yehuda Koren. Factor in the neighbors: scalable and accurate collaborative filtering. 2010](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf) which explains the concept in much more detail.

Briefly, baseline estimates look at the average rating an item earned in the entire dataset, and the average rating a user usually gives. Sometimes, some items are rated higher on average than others. In a similar way, some users rate items more critically than others. These deviations are used in baseline estimates to predict a score.

Baseline estimates can be calculated using Stochastic Gradient Descent (SGD) or Alternating Least Squares (ALS). In this example, I'll be using ALS.

I've already used `GridSearchCV` to find some optimal parameters, although it could be improved with more computational power and/or time.


```python
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.baseline_only import BaselineOnly

# param_grid = {'bsl_options': {'method': ['als'],
#                              'n_epochs': random.sample(range(10, 20), 5),
#                              'reg_u': random.sample(range(10, 30), 5),
#                              'reg_i': random.sample(range(10, 30), 5)}}

# gs = GridSearchCV(BaselineOnly, param_grid, measures=['RMSE', 'MAE'], cv=5, n_jobs = -1)
# gs.fit(data)
# print(gs.best_score['rmse'], gs.best_params['rmse'])

bsl_options = {'method': 'als', 'n_epochs': 18, 'reg_u': 11, 'reg_i': 11}

algo = BaselineOnly(bsl_options=bsl_options)
fit = algo.fit(trainset)
predictions = fit.test(testset)
accuracy.rmse(predictions, verbose=False)
```

    Estimating biases using als...
    




    1.2383393223557766



### Single Value Decomposition

Using baseline estimates are a great start, but are pretty unsophicated compared to the other models provided in the `Suprise` package.

One of the algorithms available through the `Surprise` package is a single-value decompostion (SVD) algorithm, famously used in the [Netflix competition](https://netflixprize.com/). [Popularized by Simon Funk](http://sifter.org/~simon/journal/20061211.html), this algorithm essentially boils down all a user's rating to one value, and uses that value along with a baseline estimate to predict a rating. If you want to learn more, [here's a good article](https://medium.com/@m_n_malaeb/singular-value-decomposition-svd-in-recommender-systems-for-non-math-statistics-programming-4a622de653e9)

As with the baseline estimates, I ran some cross validation to find a good combination of parameters (this took at least a full day to finish running on my laptop - I would recommend just choosing some parameters if you're just learning the `Suprise` package).

Hopefully, we'll see an improved RMSE when using this aglorithm.


```python
from surprise import SVD

# param_grid = {'n_epochs': range(5, 20),
#               'lr_all': np.linspace(0.1,0.020,9),
#               'reg_all': np.linspace(0.1,1,10)}

# gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs = -1)

# gs.fit(data)

# best RMSE score
# print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])
```


```python
algo = SVD(n_epochs=19, lr_all=0.02, reg_all=0.2)
fit = algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions, verbose=False)
```




    1.2313336546972184



We lowered the RMSE about by about 0.01. It's not a lot, but any improvement is good. However, if we wanted to re-train this model frequently, using the baseline estimate might be the more favorable option when considering the [cost of using cloud resources](https://aws.amazon.com/aml/pricing/).

## Serving Recommendations

Recommendations are no good unless you're able to serve them to customers. Depending on a business's current technology stack and business model, these recommendations can come through an e-commerce store, through an email, or even through a sales rep. Each delivery method will require different skills and tools to implement, but no matter what you'll need to get the recommendations out of your fitted algorithm and to the customer.

Luckily, there author of `Surprise` has written a function we can use to get the top-N recommendations for each user in our dataset. To avoid reinventing the wheel, I'll be using [that function](https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user).


```python
from collections import defaultdict


def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


top_n = get_top_n(predictions, n=10)
```


```python
top_n[reviews.customer_id[8]]
```




    [('B00CQ7IDZ4', 4.1219223126882296),
     ('B00HUCIOOK', 4.0721615059252416),
     ('B00INXSUKS', 3.9437460503852786),
     ('B00C7I62S2', 3.8137130798781897),
     ('B008MVVBWU', 3.6817009584164819),
     ('B008D902Q2', 3.4667934012274104)]



### Finding Popular Classes

Unfortunately, this function doesn't always recommend ten items. I suspect this is because the data is very sparse, and there is a high number of customers who have only rated one product, as well as many products that have only one rating. Regardless, we need to come up with a way to fill in the rest of the gaps.

A simple yet effective way to generate recommendations is to recommend the most popular products. This is what I'll do to generate ten unique recommendations for each user.

First, I need to build a dataset of popular products. Each person could define *popular* in a different way, but in this scenario I'm only going to consider products that have received at least 100 reviews. Then, I'll find the average rating for each product and sort them in descending order.


```python
review_count = reviews.product_id.value_counts()
review_count_ten = review_count[review_count >= 100]
hundred_reviews = reviews[reviews.product_id.isin(review_count_ten.index)]
items = (hundred_reviews[['product_id', 'star_rating'
                          ]].groupby('product_id').agg('mean').sort_values(
                              'star_rating', ascending=False).index)
```

Now, I'm going to go through each user's recommendations and add the most popular products that aren't already recommended.


```python
def recommendation_list(user_list, user_predictions, item_list):
    recommendations = {}
    for i in range(100):
        user = user_list[i]
        if user in user_predictions:
            user_recs = [
                user_predictions[user][i][0]
                for i in range(len(user_predictions[user]))
            ]
            if user_recs:
                num_items = len(user_recs)
            else:
                num_items = 0

            idx = 0
            while num_items < 10:
                product = item_list[idx]
                if product not in user_recs:
                    user_recs.append(product)
                    num_items = len(user_recs)
                idx += 1
            recommendations.update({user: user_recs})
    return recommendations


recs = recommendation_list(reviews.customer_id.unique().tolist(), top_n, items)
```


```python
example_user = reviews.customer_id.unique().tolist()[1]
recs[example_user]
```




    ['B004TB226Q',
     'B00843L74S',
     'B00EXTZ34C',
     'B001QFYKMW',
     'B005MKGPC0',
     'B0043ZWQWI',
     'B004TB2DWY',
     'B00AHAFFTO',
     'B009DRP9RU',
     'B0021AEDSM']



## Wrapping Up

Now, we have a list of recommendations that we can use to recommend new products to all of our customers! Each list of recommendations are unique to each customer, and have been generated from *real* data.

There are a few other ways to build or otherwise improve this recommender system that I avoided:

* **Use k-NN algorithms from the `Surprise` package.**

There was just too much data here. I could use a small sample of the data and build it using that, but I elected not to use it at all.

* **Use content filtering to build a hybrid recommender system.**

I've done this throughout my career, but I had access to detailed product descriptions. In this data, I only have the product title. While informative, it's not very helpful if we use TFIDF to build a content filtering algorithm.

* **Use more tuned parameters**

I did some basic cross validation to select the best parameters. If I needed to improve the recommendations, I would spend much more time tuning these parameters and make sure that the algorithm is serving the best recommendations possible with the data available.

I had a lot of fun working on the project, and learned more about the math behind recommendation system than I would have expected. Thanks for reading it all the way through, I appreciate it!

I'd love to hear way you think! Feel free to connect with me via [LinkedIn](https://www.linkedin.com/in/witkowskism/) and let me know if this was helpful!
