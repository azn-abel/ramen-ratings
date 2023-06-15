# Helping Consumers Pick the Best Instant Ramen
### By: Abel Lu

Problem Statement: The objective of this project is to identify trends in the value and ratings of instant ramen noodles over time, and to use these trends to inform average consumers about the price and quality of affordable food options they see in supermarkets. While instant ramen noodles are but a single category of affordable food, they represent a common choice that consumers make when they are looking for a cheap meal.

The data set I'm choosing to analyze for the final project is ramen-ratings.csv, which I found on kaggle: https://www.kaggle.com/datasets/residentmario/ramen-ratings?resource=download. I'm choosing it because I'm a thorough enjoyer of many likes of instant ramen. For me, anything from Samyang Buldak to Maruchan Roast Beef will get it done.

This dataset has columns for the chronologically ordered review number, name, brand, variety (product name), and style (bowl, cup or pack) of the ramen, as well as columns for the country of origin and a star rating assigned to each variety. All columns in the dataset are categorical columns with the exception of the star rating column and the review number column, which are numerical.

From this dataset, I will use the star rating as a reference to find out which countries, brands, and styles produce the highest rated ramen. Additionally, using the chronologically ordered review numbers, I will draw conclusions about how ramen ratings have changed over time.

## Loading in the dataset


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Read the csv and view it
ramen = pd.read_csv("ramen-ratings.csv")
ramen
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review #</th>
      <th>Brand</th>
      <th>Variety</th>
      <th>Style</th>
      <th>Country</th>
      <th>Stars</th>
      <th>Top Ten</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2580</td>
      <td>New Touch</td>
      <td>T's Restaurant Tantanmen</td>
      <td>Cup</td>
      <td>Japan</td>
      <td>3.75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2579</td>
      <td>Just Way</td>
      <td>Noodles Spicy Hot Sesame Spicy Hot Sesame Guan...</td>
      <td>Pack</td>
      <td>Taiwan</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2578</td>
      <td>Nissin</td>
      <td>Cup Noodles Chicken Vegetable</td>
      <td>Cup</td>
      <td>USA</td>
      <td>2.25</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2577</td>
      <td>Wei Lih</td>
      <td>GGE Ramen Snack Tomato Flavor</td>
      <td>Pack</td>
      <td>Taiwan</td>
      <td>2.75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2576</td>
      <td>Ching's Secret</td>
      <td>Singapore Curry</td>
      <td>Pack</td>
      <td>India</td>
      <td>3.75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2575</th>
      <td>5</td>
      <td>Vifon</td>
      <td>Hu Tiu Nam Vang ["Phnom Penh" style] Asian Sty...</td>
      <td>Bowl</td>
      <td>Vietnam</td>
      <td>3.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2576</th>
      <td>4</td>
      <td>Wai Wai</td>
      <td>Oriental Style Instant Noodles</td>
      <td>Pack</td>
      <td>Thailand</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2577</th>
      <td>3</td>
      <td>Wai Wai</td>
      <td>Tom Yum Shrimp</td>
      <td>Pack</td>
      <td>Thailand</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2578</th>
      <td>2</td>
      <td>Wai Wai</td>
      <td>Tom Yum Chili Flavor</td>
      <td>Pack</td>
      <td>Thailand</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2579</th>
      <td>1</td>
      <td>Westbrae</td>
      <td>Miso Ramen</td>
      <td>Pack</td>
      <td>USA</td>
      <td>0.5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2580 rows × 7 columns</p>
</div>



# Scatter Plot
First, I'll make a scatter plot of the reviews by review number and star rating to visualize the distribution.


```python
# Filter out rows with unrated ramen
ramen = ramen[ramen['Stars'] != 'Unrated']
ramen.Stars = pd.to_numeric(ramen.Stars)
stars = ramen.Stars
review_nums = ramen['Review #']
```


```python
# Create a scatter plot to visualize the data
plt.scatter(review_nums, stars, 1)

plt.title('Star Ratings of Ramen by Chronological Review Number')
plt.xlabel('Review Number')
plt.ylabel('Star Rating')

plt.show()
```


    
![png](output_6_0.png)
    


From this scatter plot, it seems that negative ramen reviews have become less frequent as time has gone on, due to the much lesser amount of 3-or-less star reviews after review number 2000 than before review number 500.

# Bar Graphs
To visualize this trend more effectively, I will make bar graph of positive and negative reviews over time.
First, I will define a **positive review** as one that has a *star rating higher than the median star rating*, which is calculated below:


```python
# Calculate the median star rating
MEDIAN_STARS = ramen.Stars.median()
print(f"Median Stars: {MEDIAN_STARS}")
```

    Median Stars: 3.75



```python
# Create 5 groups, each representing 500 reviews, then filtering them for positive reviews
group_1 = ramen[(ramen['Review #'] < 500) & (ramen.Stars > MEDIAN_STARS)]
group_2 = ramen[(ramen['Review #'] >= 500) & (ramen['Review #'] < 1000) & (ramen.Stars > MEDIAN_STARS)]
group_3 = ramen[(ramen['Review #'] >= 1000) & (ramen['Review #'] < 1500) & (ramen.Stars > MEDIAN_STARS)]
group_4 = ramen[(ramen['Review #'] >= 1500) & (ramen['Review #'] < 2000) & (ramen.Stars > MEDIAN_STARS)]
group_5 = ramen[(ramen['Review #'] >= 2000) & (ramen['Review #'] < 2500) & (ramen.Stars > MEDIAN_STARS)]

# Create a bar graph of the data
plt.bar(['0 - 499', '500 - 999', '1000 - 1499', '1500 - 1999', '2000 - 2500'],
        [len(group_1), len(group_2), len(group_3), len(group_4), len(group_5)])

# Title the graph and label the axes
plt.title('Number of Positive Reviews Grouped by Review Numbers')
plt.xlabel("Review Numbers Groups")
plt.ylabel("Number of 4+ Star Reviews")

plt.show()
```


    
![png](output_10_0.png)
    


This bar graph confirms the earlier observation that ramen star ratings have increased over time, as the sections of the bar graph representing higher review numbers (reviews later in time) have much higher bars, which represent a greater amount of positive reviews.

Next, I will define a **negative review** as one that has a *star rating lower than the median star rating*, and create a bar graph based on this definition.


```python
# Create 5 groups, each representing 500 reviews, then filtering them for regative reviews
group_1 = ramen[(ramen['Review #'] < 500) & (ramen.Stars < MEDIAN_STARS)]
group_2 = ramen[(ramen['Review #'] >= 500) & (ramen['Review #'] < 1000) & (ramen.Stars < MEDIAN_STARS)]
group_3 = ramen[(ramen['Review #'] >= 1000) & (ramen['Review #'] < 1500) & (ramen.Stars < MEDIAN_STARS)]
group_4 = ramen[(ramen['Review #'] >= 1500) & (ramen['Review #'] < 2000) & (ramen.Stars < MEDIAN_STARS)]
group_5 = ramen[(ramen['Review #'] >= 2000) & (ramen['Review #'] < 2500) & (ramen.Stars < MEDIAN_STARS)]

# Create a bar graph of the data
plt.bar(['0 - 499', '500 - 999', '1000 - 1499', '1500 - 1999', '2000 - 2500'],
        [len(group_1), len(group_2), len(group_3), len(group_4), len(group_5)])

# Title the graph and label the axes
plt.title('Number of Negative Reviews Grouped by Review Numbers')
plt.xlabel("Review Number Groups")
plt.ylabel("Number of 3.5-or-less Star Reviews")

plt.show()
```


    
![png](output_13_0.png)
    


This bar graph shows that negative reviews of ramen has decreased over time.

Next, I will create bar charts showing the number and percentage of positive reviews based on the packaging that the ramen comes in.


```python
# Separate the dataframe by the three most common packaging styles
all_cups = ramen[ramen.Style == "Cup"]
all_packs = ramen[ramen.Style == "Pack"]
all_bowls = ramen[ramen.Style == "Bowl"]

# Calculate the numbers of positive reviews per packaging style
good_cups = all_cups[ramen.Stars > MEDIAN_STARS]
good_packs = all_packs[ramen.Stars > MEDIAN_STARS]
good_bowls = all_bowls[ramen.Stars > MEDIAN_STARS]
```


```python
# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(13, 4)

# Graph number of positive reviews by style
ax1.bar(['Cup', 'Pack', 'Bowl'],
        [len(good_cups), len(good_packs), len(good_bowls)])

ax1.set_title('Number of Positive Reviews by Style')
ax1.set_xlabel("Style")
ax1.set_ylabel("Number of 4+ Star Reviews")

# Graph percentage of positive reviews by style
ax2.bar(['Cup', 'Pack', 'Bowl'],
        [len(good_cups) / len(all_cups), len(good_packs) / len(all_packs), len(good_bowls) / len(all_bowls)])

ax2.set_title('Percentage of Positive Reviews by Style')
ax2.set_xlabel("Style")
ax2.set_ylabel("% Reviews of 4+ Stars")

plt.show()
```


    
![png](output_17_0.png)
    


From these graphs, we can see that while packs are the most widely available option, bowls have a slightly higher percentage of positive ratings.

Next, I will create the same bar graph, but based on country instead of packaging style.


```python
# "USA" and "United States" both appear as countries in the dataset
ramen['Country'] = ramen['Country'].replace('USA', 'United States')
```


```python
# Creating three parallel arrays to represent the names, counts of total reviews, and counts of positive reviews for each country.
countries = ramen['Country'].unique()
countries_total_reviews = np.array([len(ramen[ramen['Country'] == country]) for country in countries])
countries_good_reviews = np.array([len(ramen[(ramen['Country'] == country) & (ramen.Stars > MEDIAN_STARS)]) for country in countries])
```


```python
# Here, I sort the parallel arrays so the countries with the highest number of positive reviews appear first.
indices = np.argsort(countries_good_reviews)

countries = countries[indices][::-1]
countries_total_reviews = countries_total_reviews[indices][::-1]
countries_good_reviews = countries_good_reviews[indices][::-1]
```


```python
# I split my graph into two subplots because it was too big
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(22, 8)

ax1.bar(countries[:19],
        countries_good_reviews[:19])

ax1.set_title('Number of Positive Reviews by Country')
ax1.set_xlabel("Country")
ax1.set_ylabel("Number of 4+ Star Reviews")

ax2.bar(countries[19:],
        countries_good_reviews[19:])

ax2.set_xlabel("Country")
ax2.set_ylabel("Number of 4+ Star Reviews")

plt.show()
```


    
![png](output_23_0.png)
    



```python
# Calculating the percentage of positive reviews and creating another parallel array for it
countries_good_percentage = countries_good_reviews / countries_total_reviews

# Sorting the parallel arrays
indices = np.argsort(countries_good_percentage)

countries = countries[indices][::-1]
countries_total_reviews = countries_total_reviews[indices][::-1]
countries_good_reviews = countries_good_reviews[indices][::-1]
countries_good_percentage = countries_good_percentage[indices][::-1]
```


```python
# Same graph as above, but for percentage of positive reviews instead of number.
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(22, 8)

ax1.bar(countries[:19],
        countries_good_reviews[:19] / countries_total_reviews[:19])

ax1.set_title('Percentage of Positive Reviews by Country')
ax1.set_xlabel("Country")
ax1.set_ylabel("Percantage of 4+ Star Reviews")

ax2.bar(countries[19:],
        countries_good_reviews[19:] / countries_total_reviews[19:])

ax2.set_xlabel("Country")
ax2.set_ylabel("Percentage of 4+ Star Reviews")

plt.show()
```


    
![png](output_25_0.png)
    


Based on the bar graphs above, we can see that while ramen from Japanese tends to be the most widely available, ramen from Brazil and Sarawak have the highest percentage of positively reviewed ramen.
