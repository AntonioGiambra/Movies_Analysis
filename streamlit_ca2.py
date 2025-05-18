# streamlit_ca2.py

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from scipy.stats import entropy

#----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    df_movies = pd.read_csv("movies.csv", encoding='latin-1')
    df_rating = pd.read_csv("rating.csv")
    df_tags = pd.read_csv("tags.csv", encoding='latin-1')
    return df_movies, df_rating, df_tags

df_movies, df_rating, df_tags = load_data()

# Title
st.title("🎬 Movie Analysis Dashboard 🎬")

# Tab Navigation
tabs = st.tabs([
    "Genre Popularity",
    "Top Movies by Genre",
    "Genre Trends Over Time",
    "Movie Similarity",
    "User Behaviour Analysis"
])

# --------------- TAB 1: Genre Popularity ------------------
with tabs[0]:
    st.header("Genre Popularity vs. Average Rating")

    df_combined = pd.merge(df_rating, df_movies, on='movieId')
    df_combined['genre_list'] = df_combined['genres'].str.split('|')
    df_exploded = df_combined.explode('genre_list')
    genre_stats = df_exploded.groupby('genre_list')['rating'].agg(['count', 'mean']).reset_index()
    genre_stats.rename(columns={'genre_list': 'genre', 'count': 'total_votes', 'mean': 'avg_rating'}, inplace=True)
    genre_stats_sorted = genre_stats.sort_values(by='total_votes', ascending=True)

    fig1 = px.scatter(
        genre_stats_sorted,
        x='total_votes',
        y='genre',
        size='avg_rating',
        color='avg_rating',
        color_continuous_scale='Blues',
        size_max=40,
        labels={'total_votes': 'Total Votes', 'genre': 'Genre', 'avg_rating': 'Average Rating'},
        title='You can see more details by hovering over each bubble.',
        hover_data={'total_votes': True, 'avg_rating': ':.2f', 'genre': True}
    )

    fig1.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        height=800,
        margin=dict(l=100, r=60, t=80, b=60),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

# --------------- TAB 2: Top Movies by Genre ------------------
with tabs[1]:
    st.header("Top 5 Movies by Genre")

    df_exploded = df_combined.explode('genre_list')
    movie_genre_stats = df_exploded.groupby(['genre_list', 'movieId', 'title']).agg(
        avg_rating=('rating', 'mean'),
        num_votes=('rating', 'count')
    ).reset_index()
    movie_genre_stats = movie_genre_stats[movie_genre_stats['num_votes'] >= 100]
    movie_genre_stats['rank_in_genre'] = movie_genre_stats.groupby('genre_list')['avg_rating']\
        .rank(method='first', ascending=False)
    top5_by_genre = movie_genre_stats[movie_genre_stats['rank_in_genre'] <= 5]
    top5_by_genre = top5_by_genre.sort_values(['genre_list', 'rank_in_genre'])
    top5_by_genre['avg_rating'] = top5_by_genre['avg_rating'].round(1)

    genres = sorted(top5_by_genre['genre_list'].unique())
    selected_genre = st.selectbox("Select a Genre:", genres)

    df_filtered = top5_by_genre[top5_by_genre['genre_list'] == selected_genre]
    df_filtered = df_filtered.sort_values(['avg_rating', 'num_votes'], ascending=[False, False])

    fig2 = px.bar(
        df_filtered,
        x='avg_rating',
        y='title',
        orientation='h',
        title=f"Top 5 Movies for {selected_genre}",
        labels={'avg_rating': 'Average Rating', 'title': 'Movie Title'},
        range_x=[0, 5],
        height=400,
        text='avg_rating',
        hover_data=['num_votes']
    )

    fig2.update_traces(
        texttemplate='%{text:.1f}',
        textposition='inside',
        marker=dict(color='rgb(21,151,221)', line=dict(color='white', width=2))
    )
    fig2.update_layout(
        title_font_size=24,
        yaxis_autorange="reversed",
        showlegend=False,
        xaxis=dict(title_font=dict(size=20)),
        yaxis=dict(title_font=dict(size=20))
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------- TAB 3: Genre Trends Over Time ------------------
with tabs[2]:
    st.header("Genre Engagement by Year")

    df1 = df_movies.copy()
    df2 = df_rating.copy()
    df1['year'] = df1['title'].str.extract(r'\((\d{4})\)').astype(float)
    df1 = df1.dropna(subset=['year'])
    df1['year'] = df1['year'].astype(int)
    df1['genres_list'] = df1['genres'].str.split('|')
    df_exploded = df1.explode('genres_list')
    df_merged = pd.merge(df2, df_exploded, on='movieId')
    genre_counts_per_year = df_merged.groupby(['year', 'genres_list']).size().reset_index(name='count')

    all_years = sorted(df_merged['year'].unique())
    all_genres = sorted(df_exploded['genres_list'].unique())
    full_index = pd.MultiIndex.from_product([all_years, all_genres], names=['year', 'genres_list'])
    genre_counts_per_year_full = genre_counts_per_year.set_index(['year', 'genres_list']).reindex(full_index, fill_value=0).reset_index()

    colours = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF",
               "#00FFFF", "#FFA500", "#800080", "#DC143C", "#4682B4",
               "#DAA520", "#20B2AA", "#9932CC", "#D2B48C", "#7B68EE",
               "#8B0000", "#008080", "#FF69B4", "#CD853F", "#A0522D"]
    genre_list = sorted(genre_counts_per_year_full['genres_list'].unique())
    color_map = {genre: colours[i % len(colours)] for i, genre in enumerate(genre_list)}

    fig3 = px.bar(
        genre_counts_per_year_full,
        y='genres_list',
        x='count',
        color='genres_list',
        animation_frame='year',
        orientation='h',
        range_x=[0, genre_counts_per_year_full['count'].max() * 1.1],
        color_discrete_map=color_map,
        labels={'genres_list': 'Genre', 'count': 'Ratings'}
    )
    fig3.update_layout(
        xaxis=dict(title_font=dict(size=20)),
        yaxis=dict(title_font=dict(size=20)),
        height=800,
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

# --------------- TAB 4: Movie Similarity ------------------
with tabs[3]:
    st.header("Top 10 Similar Movies by Genre")

    df_movies['genres_list'] = df_movies['genres'].apply(lambda x: x.split('|'))
    df_movies['year'] = df_movies['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None)
    df_movies['title_clean'] = df_movies['title'].apply(lambda x: re.sub(r'\s\(\d{4}\)', '', x))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df_movies['genres_list'])
    cosine_sim = cosine_similarity(genre_matrix)

    def get_top_similar_movies(title, top_n=10):
        if title not in df_movies['title'].values:
            st.warning(f"'{title}' not found.")
            return pd.DataFrame()
        idx = df_movies[df_movies['title'] == title].index[0]
        sim_scores = cosine_sim[idx]
        sim_df = pd.DataFrame({
            'title': df_movies['title'],
            'similarity': sim_scores
        })
        sim_df = sim_df[sim_df['title'] != title]
        sim_df = sim_df.sort_values(by='similarity', ascending=False).head(top_n)
        sim_df['similarity_percent'] = sim_df['similarity'] * 100
        return sim_df

    selected_movie = st.selectbox("Choose a movie:", sorted(df_movies['title'].unique()))
    sim_df = get_top_similar_movies(selected_movie)

    if not sim_df.empty:
        fig4 = px.bar(
            sim_df,
            x='similarity_percent',
            y='title',
            orientation='h',
            range_x=[0, 100],
            text=sim_df['similarity_percent'].round(1),
            labels={'similarity_percent': 'Similarity (%)', 'title': 'Movie Title'},
            title=f"Top 10 Similar Movies to '{selected_movie}'"
        )
        fig4.update_traces(
            texttemplate='<b>%{text:.1f}%</b>',
            textposition='inside',
            marker=dict(color='rgb(21,151,221)', line=dict(color='white', width=2))
        )
        fig4.update_layout(
            title_font_size=24,
            yaxis_autorange='reversed',
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            margin=dict(l=150, t=80),
            height=500
        )
        st.plotly_chart(fig4, use_container_width=True)

# --------------- TAB 5: User Profiles ------------------
with tabs[4]:
    st.header("User Behaviour Analysis")

    # Copy original dataframes
    df3 = df_movies.copy()
    df4 = df_rating.copy()

    # Split genres string into lists
    df3['genres'] = df3['genres'].str.split('|')

    # Merge ratings with movie metadata
    df_merged = df4.merge(df3, on='movieId')

    # Compute user profile metrics
    user_profiles = []
    for user_id, group in df_merged.groupby('userId'):
        ratings = group['rating']
        genres_list = group['genres'].explode()
        genre_distribution = genres_list.value_counts(normalize=True)

        user_profile = {
            'userId': user_id,
            'Average Rating': ratings.mean(),
            'Rating Std Dev': ratings.std(ddof=0),
            'Total Ratings': len(ratings),
            '% High Ratings': (ratings >= 4.0).mean() * 100,
            'Genre Diversity': entropy(genre_distribution, base=2)
        }
        user_profiles.append(user_profile)

    # Create user profile DataFrame
    df_users = pd.DataFrame(user_profiles)

    # Define selected features for radar chart
    features = ['Average Rating', 'Rating Std Dev', 'Total Ratings', '% High Ratings', 'Genre Diversity']
    angles = features + [features[0]]  # Add the first feature at the end to close the radar shape

    # Normalize selected features
    scaler = MinMaxScaler()
    df_scaled = df_users.copy()
    df_scaled[features] = scaler.fit_transform(df_users[features])

    # Dropdown to select user
    selected_user = st.selectbox("Select User ID:", df_scaled['userId'].astype(int))

    # Get both scaled and original values for the selected user
    user_scaled_row = df_scaled[df_scaled['userId'] == selected_user].iloc[0]
    user_original_row = df_users[df_users['userId'] == selected_user].iloc[0]

    # Scaled values for plotting
    scaled_values = user_scaled_row[features].tolist() + [user_scaled_row[features[0]]]

    # Original values for hover text
    original_values = user_original_row[features].tolist() + [user_original_row[features[0]]]

    # Custom hover text using original (non-scaled) values
    hovertext = [
        f"{label}: {val:.2f}%" if label == '% High Ratings' else f"{label}: {val:.2f}"
        for label, val in zip(angles, original_values)
    ]

    # Create radar chart with Plotly
    fig5 = go.Figure(data=go.Scatterpolar(
        r=scaled_values,
        theta=angles,
        fill='toself',
        name=f"User {int(selected_user)}",
        hovertext=hovertext,         # Display original values on hover
        hoverinfo='text'
    ))

    # Customize layout
    fig5.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],         # Keep normalized scale for visualization
                showticklabels=False  # Hide tick labels as requested
            ),
            angularaxis=dict(
                tickfont=dict(size=16)
            )
        ),
        title=dict(
            text=f"User {int(selected_user)} Profile",
            font=dict(size=24),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=700
    )

    # Display chart in Streamlit
    st.plotly_chart(fig5, use_container_width=True)
