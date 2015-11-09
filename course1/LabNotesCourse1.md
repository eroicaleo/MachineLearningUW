# Week 1 Lab

**Basic command**

`import graphlab`, `graphlab.SFrame("filename")`
`sf.head()`, `sf.tail()`

Canvas command: `sf.show()` is mad cool!

It opens in new tab in the browser, we can also do this:
```python
graphlab.canvas.set_target('ipynb')
sf['age'].show(view='Categorical')
```

**Interacting with columns in SFrame**
```python
sf['Country']
sf['age'].mean()
sf['Full Name'] = sf['First Name'] + sf['Last Name']
sf['age'] * sf['age']
```

**Using .apply() for data transformation**

```python
sf['Country'].apply(transform_country)
```

# Week 2 Lab

```python
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")
train_data, test_data = sales.random_split(.8, seed=0)
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'])
print test_data['price'].mean()
print sqft_model.evaluate(test_data)
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(test_data['sqft_living'], test_data['price'], '.',
        test_data['sqft_living'], sqft_model.predict(test_data), '_')

sqft_model.get('coefficients')
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

my_features_model = graphlab.linear_regression.create(train_data, target='price', features=my_features)
print my_features_model.evaluate(test_data)
house1 = sales[sales['id'] == '5309101200']
print sqft_model.predict(house1)
```

# Week 3 Lab

```python
products['rating'].show(view = 'Categorical')
# ignore 3 stars
products = products[products['rating'] != 3]
# positive sentiment
products['sentiment'] = products['rating'] >= 4
train_data, test_data = products.random_split(.8, seed=0)
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show(view='Evaluation')
# change the threshold by sliding the button.
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews
, output_type='probability')
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews[-1]['review']
```

# Week 4 Lab

## Find the length of a SFrame

```python
len(people)
```

## Select specific people

```python
obama = people[people['name'] == 'Barack Obama']
clooney = people[people['name'] == 'George Clooney']
clooney['text']
```

## Get the word counts for Obama article

```python
obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])
```

## Sort the word counts for the Obama article

```python
# Make it looks better
# Note that the [[]]
# >>> type(obama[['word_count']])
# graphlab.data_structures.sframe.SFrame
# >>> type(obama['word_count'])
# graphlab.data_structures.sarray.SArray
obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word', 'count'])

obama_word_count_table.sort('count', ascending=False)
```

## Compute TF-IDF for the corpus

```python
# word count for all people
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf['docs']

## Examine the TF-IDF for the Obama article
obama = people[people['name'] == 'Barack Obama']
obama[['tfidf']].stack('tfidf', new_column_name = ['word', 'tfidf']).sort('tfidf', ascending=False)
```

## Manually compute distances between a few people

```python
graphlab.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])
graphlab.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])
```

## Build a nearest neighbor model for document retrieval

```python
knn_model = graphlab.nearest_neighbors.create(people, features=['tfidf'], label='name')
knn_model.query(obama)
```

# Week 5

## Set the canvas to notebook

```python
graphlab.canvas.set_target('ipynb')
# Show the bar chart
song_data['song'].show()
```

## Count the unique users

```python
users = song_data['user_id'].unique()
len(users)
```

## Create a song recommender
```python
train_data, test_data = song_data.random_split(.8, seed=0)
```

### Simple popularity-based recommender

```python
# Training
popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')
# Prediction
popularity_model.recommend(users=[users[0]])
```

### Build a song recommender with personalization

```python
# Training
personalized_mode = graphlab.item_similarity_recommender.create(train_data,
                                                               user_id='user_id',
                                                               item_id='song')
# Applying the personalized model to make song recommendation
personalized_mode.recommend(users=[users[0]])

# Get similar items
personalized_mode.get_similar_items(['With Or Without You - U2'])
personalized_mode.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])
```

### Quantitative comparison between the models

```python
%matplotlib inline
model_performance = graphlab.recommender.util.compare_models(test_data,
                                                            [popularity_model, personalized_mode],
                                                            user_sample=0.05)
```

### Sort the artist by their popularity (total number of songs played)

```python
song_data.groupby(key_columns='artist', operations=
{'total_count': graphlab.aggregate.SUM('listen_count')}).sort('total_count',
ascending=False)
```

### Sort the song by their popularity (total number of songs recommended)
```python
recommendations.groupby(key_columns='song',
operations={'count': graphlab.aggregate.COUNT()}).sort('count', ascending=False)
```

# Week 6 Lab

## Train/test a classifier on the raw image pixels

```python
raw_pixel_model = graphlab.logistic_classifier.create(image_train, target='label',
                                                     features=['image_array'])
raw_pixel_model.predict(image_test[0:3])
raw_pixel_model.evaluate(image_test)
```

## Load deep feature model and convert data set raw features to deep features

```python
deep_learning_model = graphlab.load_model('imagenet_model')
image_train['deep_features'] = deep_learning_model.extract_features(image_train)
```

## Use the deep features with logistic regression

```python
deep_feature_model = graphlab.logistic_classifier.create(image_train,
target='label', features=['deep_features'])
deep_feature_model.predict(image_test[0:3])
deep_feature_model.evaluate(image_test)
```

## Use the deep features with knn

```python
knn_model = graphlab.nearest_neighbors.create(image_train, features=['deep_features'], label='id')
# Note here, we have to use [18:19] to return a SFrame object.
cat = image_train[18:19]
knn_model.query(cat)
# Note the filter_by function
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'], 'id')
cat_neighbors = get_images_from_ids(knn_model.query(cat))
cat_neighbors['image'].show()
car = image_train[8:9]
get_images_from_ids(knn_model.query(car))['image'].show()
# Define a lambda to make show easier
show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()
show_neighbors(8)
```
