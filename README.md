
# Generating Word Embeddings - Lab

## Introduction

In this lab, you'll learn how to generate word embeddings by training a Word2Vec model, and then embedding layers into Deep Neural Networks for NLP!

## Objectives

You will be able to:

* Demonstrate a basic understanding of the architecture of the Word2Vec model
* Demonstrate an understanding of the various tunable parameters of Word2Vec such as vector size and window size

## Getting Started

In this lab, you'll start by creating your own word embeddings by making use of the Word2Vec Model. Then, you'll move onto building Neural Networks that make use of **_Embedding Layers_** to accomplish the same end-goal, but directly in your model. 

As you've seen, the easiest way to make use of Word2Vec is to import it from the [Gensim Library](https://radimrehurek.com/gensim/). This model contains a full implementation of Word2Vec, which you can use to begin training immediately. For this lab, you'll be working with the [News Category Dataset from Kaggle](https://www.kaggle.com/rmisra/news-category-dataset/version/2#_=_).  This dataset contains headlines and article descriptions from the news, as well as categories for which type of article they belong to.

Run the cell below to import everything you'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
from gensim.models import Word2Vec
from nltk import word_tokenize
```

    C:\Users\FlatIron_User\.conda\envs\learn-env\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

Now, import the data. The data stored in the file `'News_Category_Dataset_v2.json'`.  This file is compressed, so that it can be more easily stored in a github repo. **_Make sure to unzip the file before continuing!_**

In the cell below, use the `read_json` function from pandas to read the dataset into a DataFrame. Be sure to include the parameter `lines=True` when reading in the dataset!

Once you've loaded in the data, inspect the head of the DataFrame to see what your data looks like. 


```python
raw_df = pd.read_json('~/DataScience/News_Category_Dataset_v2.json', lines=True)
raw_df.head()
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
      <th>authors</th>
      <th>category</th>
      <th>date</th>
      <th>headline</th>
      <th>link</th>
      <th>short_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Melissa Jeltsen</td>
      <td>CRIME</td>
      <td>2018-05-26</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Andy McDonald</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Jim Carrey Blasts 'Castrato' Adam Schiff And D...</td>
      <td>https://www.huffingtonpost.com/entry/jim-carre...</td>
      <td>The actor gives Dems an ass-kicking for not fi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ron Dicker</td>
      <td>ENTERTAINMENT</td>
      <td>2018-05-26</td>
      <td>Julianna Margulies Uses Donald Trump Poop Bags...</td>
      <td>https://www.huffingtonpost.com/entry/julianna-...</td>
      <td>The "Dietland" actress said using the bags is ...</td>
    </tr>
  </tbody>
</table>
</div>



## Preparing the Data

Since you're working with text data, you need to do some basic preprocessing including tokenization. Notice from the data sample that two different columns contain text data--`headline` and `short_description`. The more text data your Word2Vec model has, the better it will perform. Therefore, you'll want to combine the two columns before tokenizing each comment and training your Word2Vec model. 

In the cell below:

* Create a column called `combined_text` that consists of the data from `df.headline` plus a space character (`' '`) plus the data from `df.short_description`.
* Use the `combined_text` column's `map()` function and pass in `word_tokenize`. Store the result returned in `data`.


```python
raw_df['combined_text'] = raw_df.headline + ' ' +  raw_df.short_description
data = raw_df.combined_text.map(word_tokenize)

```

Inspect the first 5 items in `data` to see how everything looks. 


```python
data[:5]
```




    0    [There, Were, 2, Mass, Shootings, In, Texas, L...
    1    [Will, Smith, Joins, Diplo, And, Nicky, Jam, F...
    2    [Hugh, Grant, Marries, For, The, First, Time, ...
    3    [Jim, Carrey, Blasts, 'Castrato, ', Adam, Schi...
    4    [Julianna, Margulies, Uses, Donald, Trump, Poo...
    Name: combined_text, dtype: object



Notice that although the words are tokenized, they are still in the same order they were in as headlines. This is important, because the words need to be in their original order for Word2Vec to establish the meaning of them. Remember that for a Word2Vec model you can specify a  **_Window Size_** that tells the model how many words to take into consideration at one time. 

If your window size was 5, then the model would start by looking at the words "Will Smith joins Diplo and", and then slide the window by one, so that it's looking at "Smith joins Diplo and Nicky", and so on, until it had completely processed the text example at index 1 above. By doing this for every piece of text in the entire dataset, the Word2Vec model learns excellent vector representations for each word in an **_Embedding Space_**, where the relationships between vectors capture semantic meaning (recall the vector that captures gender in the previous "king - man + woman = queen" example you saw).

Now that you've prepared the data, train your model and explore a bit!

## Training the Model

Start by instantiating a Word2Vec Model from gensim below. 

In the cell below:

* Create a `Word2Vec` model and pass in the following arguments:
    * The dataset we'll be training on, `data`
    * The size of the word vectors to create, `size=100`
    * The window size, `window=5`
    * The minimum number of times a word needs to appear in order to be counted in  the model, `min_count=1`.
    * The number of threads to use during training, `workers=4`


```python
model = Word2Vec(data, size=100, window=5, min_count=1, workers=4)
```

Now, that you've instantiated Word2Vec model, train it on your text data. 

In the cell below:

* Call `model.train()` and pass in the following parameters:
    * The dataset we'll be training on, `data`
    * The `total_examples`  of sentences in the dataset, which you can find in `model.corpus_count`. 
    * The number of `epochs` you want to train for, which we'll set to `10`


```python
model.train(data, total_examples=model.corpus_count, epochs=10)
```




    (55562985, 67352790)



Great! you now have a fully trained model! The word vectors themselves are stored inside of a `Word2VecKeyedVectors` instance, which is stored inside of `model.wv`. To simplify this, restore this object inside of the variable `wv` to save yourself some keystrokes down the line. 


```python
wv = model.wv
```

## Examining Your Word Vectors

Now that you have a trained Word2Vec model, go ahead and explore the relationships between some of the words in the corpus! 

One cool thing you can use Word2Vec for is to get the most similar words to a given word. You can do this passing in the word to `wv.most_similar()`. 

In the cell below, try getting the most similar word to `'Texas'`.


```python
wv.most_similar('Texas')
```




    [('Louisiana', 0.8097705841064453),
     ('Ohio', 0.8076143264770508),
     ('Maryland', 0.8057453036308289),
     ('Arkansas', 0.8051851987838745),
     ('Illinois', 0.8040783405303955),
     ('Pennsylvania', 0.8036627769470215),
     ('Oklahoma', 0.7925784587860107),
     ('Arizona', 0.7790142893791199),
     ('California', 0.7771258354187012),
     ('Connecticut', 0.7735964059829712)]



Interesting! All of the most similar words are also states. 

You can also get the least similar vectors to a given word by passing in the word to the `most_similar()` function's `negative` parameter. 

In the cell below, get the least similar words to `'Texas'`.


```python
wv.most_similar(negative='Texas')
```




    [('went-off', 0.462734580039978),
     ('Likened', 0.40024253726005554),
     ('Reformist', 0.3914555311203003),
     ('memory-loss', 0.38415995240211487),
     ('Sis-In-Law', 0.37695997953414917),
     ('unachievable', 0.37442946434020996),
     ('Parent/Grandparent', 0.3691348135471344),
     ('g.o.b', 0.36242997646331787),
     ('Ex-hockey', 0.36147189140319824),
     ('Headstrong', 0.36068400740623474)]



This seems like random noise. It is a result of the way Word2Vec is computing the similarity between word vectors in the embedding space. Although the word vectors closest to a given word vector are almost certainly going to have similar meaning or connotation with your given word, the word vectors that the model considers 'least similar' are just the word vectors that are farthest away, or have the lowest cosine similarity. It's important to understand that while the closest vectors in the embedding space will almost certainly share some level of semantic meaning with a given word, there is no guarantee that this relationship will hold at large distances. 

You can also get the vector for a given word by passing in the word as if you were passing in a key to a dictionary. 

In the cell below, get the word vector for `'Texas'`.


```python
wv['Texas']
```




    array([-0.01194934,  0.0874939 , -0.27812955,  0.3104571 ,  0.58127207,
           -2.4123058 , -0.56134826, -0.5016424 , -0.5712268 ,  1.0780897 ,
            1.7385156 ,  1.994228  ,  1.0927333 ,  0.70763195, -0.50771755,
           -0.08227679, -1.2751693 , -0.29554215, -0.68491757,  0.6317347 ,
            1.2252463 ,  0.5768887 , -0.7625231 , -2.9972115 , -1.1046884 ,
            0.80726534, -0.22146553, -0.2912137 ,  1.3038353 , -0.3506807 ,
           -0.31796038,  0.95840496, -0.10306006,  2.0717647 ,  0.7552676 ,
            0.85939384, -0.11264554,  1.1239612 , -1.1003317 , -0.3229932 ,
            0.6595538 ,  0.07685817, -0.02762237, -0.86376816,  1.8733982 ,
            1.0914831 , -2.7195835 ,  0.1072415 ,  2.136617  ,  1.7636586 ,
           -1.2838198 , -0.76125413,  1.7742596 , -0.21572663, -1.2594517 ,
            1.1940857 , -2.404049  ,  0.00709415,  1.6379461 , -2.6633637 ,
            0.802656  , -1.0378512 , -0.89877474,  3.8078027 ,  0.789198  ,
           -0.20152748, -1.1239738 ,  2.5157225 , -0.40786085, -2.077488  ,
           -1.3285271 ,  0.61633915, -0.6682665 ,  2.8367336 ,  1.3289468 ,
           -0.06233075, -1.4289793 , -1.3266212 , -1.0146928 , -0.326395  ,
            1.1911641 , -0.3109972 , -2.8573003 ,  2.1149867 , -0.31896284,
            1.6039978 ,  0.1684422 ,  0.15242776,  1.9532567 , -0.07461026,
            0.96116227,  1.0381123 , -0.9644637 , -1.1093541 ,  1.2024145 ,
            0.750028  , -1.926398  ,  3.043268  , -0.1200004 ,  0.7398114 ],
          dtype=float32)



Now get all of the word vectors from the object at once. You can find these inside of `wv.vectors`. Try it out in the cell below.  


```python
wv.vectors
```




    array([[ 0.6675973 , -0.02643064, -0.76537037, ...,  0.8402526 ,
            -0.27655014,  1.1173735 ],
           [ 1.5937591 , -1.188032  , -2.2450347 , ..., -1.1419069 ,
            -1.7836182 , -0.290631  ],
           [-2.053545  , -0.16112214, -1.842231  , ...,  0.2100848 ,
            -2.021004  ,  0.8644676 ],
           ...,
           [-0.02158028, -0.01714127, -0.00907745, ..., -0.09200953,
            -0.02436004,  0.16228808],
           [-0.01585435, -0.06191238, -0.04790782, ...,  0.02827319,
            -0.01817709,  0.05311523],
           [ 0.06181476, -0.0376799 ,  0.02677813, ..., -0.0840729 ,
            -0.01609733,  0.06780389]], dtype=float32)



As a final exercise, try to recreate the _'king' - 'man' + 'woman' = 'queen'_ example previously mentioned. You can do this by using the `most_similar` function and translating the word analogies into an addition/subtraction formulation (as shown above). Pass the original comparison, which you are calculating a difference between, to the negative parameter, and the analogous starter you want to apply the same transformation to, to the `positive` parameter.

Do this now in the cell below. 


```python
wv.most_similar(positive=['king', 'woman'], negative=['man'])
```




    [('princess', 0.6162869930267334),
     ('symbol', 0.6058436036109924),
     ('queen', 0.6008519530296326),
     ('dancer', 0.5948484539985657),
     ('lover', 0.5922095775604248),
     ('villain', 0.5836859941482544),
     ('purveyor', 0.5797420740127563),
     ('title', 0.56638503074646),
     ('unicorn', 0.5644869804382324),
     ('fan', 0.5587800145149231)]



As you can see from the output above, your model isn't perfect, but 'Queen' is still in the top 3, and with 'Princess' not too far behind. As you can see from the word in first place, 'reminiscent', your model is far from perfect. This is likely because you didn't have enough training data. That said, given the small amount of training data provided, the model still performs remarkably well! 

In the next lab, you'll reinvestigate transfer learning, loading in the weights from an open-sourced model that has already been trained for a very long time on a massive amount of data. Specifically, you'll work with the GloVe model from the Stanford NLP Group. There's not really any benefit from training the model ourselves, unless your text uses different, specialized vocabulary that isn't likely to be well represented inside an open-source model.

## Summary

In this lab, you learned how to train and use a Word2Vec model to created vectorized word embeddings!
