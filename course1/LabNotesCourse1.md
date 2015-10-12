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
