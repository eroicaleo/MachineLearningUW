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
