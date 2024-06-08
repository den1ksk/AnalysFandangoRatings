import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read csv.
all_sites = pd.read_csv('data/all_sites_scores.csv')
fandango = pd.read_csv('data/fandango_scrape.csv')

# FANDANGO RATINGS.

# Сreating graph of fandango ratings and votes.
plt.figure(figsize=(8, 4), dpi=150)

sns.scatterplot(data=fandango, x='RATING', y='VOTES', label='Fandango Ratings', color='red', alpha=0.7)
plt.title('Ratings and Votes', fontsize=14, fontweight='bold')
plt.legend(loc=(1, 0.5))

# Save the graph.(uncomment to save)
# plt.savefig('results/fandango_ratings_and_votes.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# Create column 'year' from 'FILM' column.
fandango['YEAR'] = fandango['FILM'].str.extract(r'\((\d{4})\)$')

# Check how many films were released in each year.
print(fandango['YEAR'].value_counts())


# Сreating graph of number of films per year.
plt.figure(figsize=(12, 6), dpi=150)

ax = sns.countplot(data=fandango, x='YEAR', hue='YEAR', palette='viridis')
ax.set_title('Number of films per year', fontsize=14, fontweight='bold')
ax.legend(labels=fandango['YEAR'].unique())
ax.set_yscale('log')

# Save the graph.(uncomment to save)
# plt.savefig('results/number_of_films_per_year.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# filtered movies with voices.
filtered_movies = fandango[fandango['VOTES'] > 0]


# Сreating graph of distribution of ratings.
plt.figure(figsize=(8, 4), dpi=150)

sns.kdeplot(data=filtered_movies, x='RATING', fill=True, label='True Rating')
sns.kdeplot(data=filtered_movies, x='STARS', fill=True, label='Stars Displayed')

plt.title('Distribution of Ratings', fontsize=14, fontweight='bold')
plt.xlim(0, 5)
plt.legend(loc='upper left')

# Save the graph.(uncomment to save)
# plt.savefig('results/distribution_of_ratings.png', bbox_inches='tight', dpi=300, format='png')

plt.show()


# Сreating a copy of the df to avoid the SettingWithCopyWarning.
filtered_movies = fandango[fandango['VOTES'] > 0].copy()

# Find the difference in values.
filtered_movies.loc[:, 'STARS_DIFF'] = (filtered_movies['STARS'] - filtered_movies['RATING']).round(1)
# Variance output.
print(filtered_movies['STARS_DIFF'])


# Сreating graph of differences between STAR and RATING.
plt.figure(figsize=(8, 4), dpi=150)

sns.countplot(data=filtered_movies, x='STARS_DIFF', palette='viridis', hue='STARS_DIFF')
plt.title('Differences between STAR and RATING', fontsize=14, fontweight='bold')

# Save the graph.(uncomment to save)
# plt.savefig('results/differences_between_star_and_rating.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# The chart shows one difference in 1.0. Find this film.
x = filtered_movies[filtered_movies['STARS_DIFF'] == 1.0]
# print(x): 381 Turbo Kid (2015) 5.0 4.0 2 2015 1.0


# Rating by Rotten Tomatoes.


# Сreating graph rating by 'rotten tomatoes'.
plt.figure(figsize=(8, 4), dpi=150)

sns.scatterplot(data=all_sites, x='RottenTomatoes', y='RottenTomatoes_User')
plt.title('Rating by Rotten Tomatoes', fontsize=14, fontweight='bold')

# Save the graph.(uncomment to save)
# plt.savefig('results/rating_by_rotten_tomatoes.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# Calculate the difference between critics' and users' ratings on Rotten Tomatoes.
all_sites['Rotten_Diff'] = all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']

# Positive values - critics rated higher, negative values - users rated higher.
avg = all_sites['Rotten_Diff'].apply(abs).mean()
# avg absolute difference between ratings from critics and user ratings.
# print(avg) : 15.095890410958905.


# Creating graph distribution of the difference between critics' and users' ratings.
plt.figure(figsize=(8, 4), dpi=150)

sns.histplot(data=all_sites, x='Rotten_Diff', kde=True, bins=20)
plt.title('RT Difference between Critics and Users', fontsize=14, fontweight='bold')

# Save the graph.(uncomment to save)
# plt.savefig('results/rt_difference_between_critics_and_users.png', bbox_inches='tight', dpi=300, format='png')

plt.show()


# Top 5 movies that are highest rated by users, compared to ratings from critics.
top5_mall = all_sites.nsmallest(5, 'Rotten_Diff')[['FILM', 'Rotten_Diff']]
print("Users Love but Critics Hate", '\n', top5_mall)

# Top 5 movies that are highest rated by critics, compared to ratings from users.
top5_high = all_sites.nlargest(5,  'Rotten_Diff')[['FILM',  'Rotten_Diff']]
print("Critics Love, but Users Hate",  '\n', top5_high)


# Rating by MetaCritic.

# Creating graph rating by 'MetaCritic'.
plt.figure(figsize=(8, 4), dpi=150)

sns.scatterplot(data=all_sites, x='Metacritic', y='Metacritic_User')
plt.title('Rating by MetaCritic', fontsize=14, fontweight='bold')

# Save the graph.(uncomment to save)
# plt.savefig('results/rating_by_metacritic.png', bbox_inches='tight', dpi=300, format='png')

plt.show()


# Rating by IMDB.

# Creating graph rating by 'IMDB'.
plt.figure(figsize=(8, 4), dpi=150)

sns.scatterplot(data=all_sites, x='Metacritic_user_vote_count', y='IMDB_user_vote_count')
plt.title('Rating by IMDB', fontsize=14, fontweight='bold')

# Save the graph.(uncomment to save)
# plt.savefig('results/rating_by_imdb.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# Selecting the movie that received the most votes on imdb.
imdb_max = all_sites.nlargest(1, 'IMDB_user_vote_count')
# print(imdb_max): The Imitation Game (2014).

# Selecting the movie that received the most votes on MetaCritic.
metacritic_max = all_sites.nlargest(1,  'Metacritic_user_vote_count')
# print(metacritic_max) # Mad Max: Fury Road (2015).


# Comparing Fandango's ratings with those of other companies.

# Merging tables.
df = pd.merge(fandango, all_sites, on='FILM', how='inner')
print(df.info())
print(df.head())

# Change of rating scale.

normalization_parameters = [
    ('RottenTomatoes', 20, 'RT_Norm'),
    ('RottenTomatoes_User', 20, 'RTU_Norm'),
    ('Metacritic', 20, 'Meta_Norm'),
    ('Metacritic_User', 2, 'Meta_U_Norm'),
    ('IMDB', 2, 'IMDB_Norm')
]

# Check if the normalized columns already exist before creating them.

for col, divisor, new_col in normalization_parameters:
    if new_col not in df.columns:
        df[new_col] = np.round(df[col] / divisor, 1)

# Creating new dataframe with normalized ratings.
norm_scores = df[['STARS', 'RATING', 'RT_Norm', 'RTU_Norm', 'Meta_Norm', 'Meta_U_Norm', 'IMDB_Norm']]


# Comparison of rating distributions from different companies.

# Creating graph rotten tomatoes vs fandango.
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

sns.kdeplot(data=norm_scores['RT_Norm'], clip=[0, 5], ax=ax, fill=True, label='Rotten Tomatoes')
sns.kdeplot(data=norm_scores['STARS'], clip=[0, 5], ax=ax, fill=True, label='Fandango')

ax.set_title('Rotten Tomatoes vs Fandango', fontsize=14, fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Stars')

ax.legend(loc='upper left')
ax.set_xlim(0, 5)

# Save the graph.(uncomment to save)
# plt.savefig('results/fandango_vs_rotten_tomatoes.png', bbox_inches='tight', dpi=300)

plt.show()

# Creating graph comparing all normalized ratings.
plt.subplots(figsize=(8, 4), dpi=150)

sns.histplot(data=norm_scores, bins=50)
plt.title('Rating Distributions Across Different Companies', fontsize=14, fontweight='bold')

plt.ylabel('Count')
plt.xlabel('Normalized Rating')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the graph.(uncomment to save)
# plt.savefig('results/rating_distributions.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# Creating graph comparing ratings from different companies.
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

# Loop for adding data and labels to the legend.
for column in norm_scores.columns:
    sns.kdeplot(data=norm_scores[column], clip=[0, 5], fill=True, ax=ax, label=column)

ax.set_title('Distributions of ratings from different companies', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(0,  5)

# Save the graph.(uncomment to save)
# plt.savefig('results/rating_distributions_from_diff_comp.png', bbox_inches='tight', dpi=300, format='png')

plt.show()


# Fandango's ratings are clearly higher than those of other companies,
# especially when you look at the rounded up ratings values.


# Search for the movie that is most different.

# Creating a new pd.
norm_films = df[['STARS', 'RATING', 'RT_Norm', 'RTU_Norm', 'Meta_Norm', 'Meta_U_Norm', 'IMDB_Norm', 'FILM']]

# Creating graph Ratings for RT Critic's 10 Worst Reviewed Films.
plt.figure(figsize=(12,  6), dpi=150)

# Selects 10 films with lowest 'RT_Norm' ratings, drops 'FILM' column.
worst_films = norm_films.nsmallest(10, 'RT_Norm').drop('FILM', axis=1)

sns.kdeplot(data=worst_films, clip=[0, 5], fill=True, palette='Set1')

plt.title('Ratings for RT Critics 10 Worst Reviewed Films',  fontsize=14, fontweight='bold')
plt.xlim(0, 5)
plt.xlabel('Rating')

# Save the graph.(uncomment to save)
# plt.savefig('results/rt_critic_worst_films.png', bbox_inches='tight', dpi=300, format='png')

plt.show()

# The graph shows that there is a movie with a huge difference.
# Let's find him.

print(norm_films.nsmallest(10, 'RT_Norm'))
# There is a movie number 25 with a 4.5 rating. FILM: Taken 3 (2015)

print(norm_films.iloc[25])
# Let's find the average rating (without Fandango)
# 0.4+2.3+1.3+2.3+3 = 9.3; 
# 9.3/5 = 1.86 avg rating





