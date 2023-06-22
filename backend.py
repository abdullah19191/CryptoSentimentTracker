import praw
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import re
from streamlit import cache
from config import (
    bitcoin_refs,
    ether_refs,
    xrp_refs,
    binance_refs,
    litecoin_refs,
    cardano_refs,
    dogecoin_refs,
    polkadot_refs,
    chainlink_refs,
    stellar_refs,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

subreddit_names = [
    "cryptocurrency",
    "Bitcoin",
    "Crypto",
    "ethereum",
    "CryptoCurrencyTrading",
    "CryptoMarkets",
    "NFT",
    "Altcoin",
    "CryptoWallets",
    "Binance",
]
# subreddit_names = ["BTC", "ETC", "Binance", "LTC"]
posts = set()

# st.cache_data.clear()


# @st.cache_data(persist="disk")
def authenticate_reddit():
    # Reddit API authentication code
    try:
        reddit = praw.Reddit(
            client_id="TlMkAZVd8CO4k1j54fbcQg",
            client_secret="EbPZRCCcmXXkQbkzo6BFSrkLNYQdFg",
            user_agent="CryptoDashboard990 by /u/DavidMi180",
            password="Pakistan180",
            username="DavidMi180",
            # client_id="AFQf5b4mzQjv9lgoZ7Q_5A",
            # client_secret="_luV8kAyWuyQNCMO5d20V5hXe9xd2g",
            # user_agent="Dev by /u/DavidMi990",
            # password="Pakistan990",
            # username="DavidMi990",
        )
        return reddit
    except praw.exceptions.PRAWException as e:
        # Handle authentication error
        print("Failed to authenticate with Reddit API:", str(e))
        return None


# @st.cache_data(persist="disk", show_spinner=False)
def fetch_reddit_posts(_reddit, num_posts):
    # Fetching and processing Reddit posts code
    try:
        posts = set()  # Initialize the set to store posts
        # subreddit_names = ['crypto', 'ethereum', 'ripple','BTC','ETC']  # Example subreddit names
        # Fetching and processing Reddit posts code
        for subreddit_name in subreddit_names:
            subreddit = _reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=num_posts):
                if post.title and post.score:
                    posts.add(
                        frozenset(
                            {
                                "Title": post.title,
                                "Score": post.score,
                                "Timestamp": datetime.datetime.fromtimestamp(
                                    post.created_utc
                                ),
                            }.items()
                        )
                    )

        # Converting the set of frozensets to a list of dictionaries
        posts_list = [dict(post) for post in posts]
        reddit_df = pd.DataFrame.from_dict(posts_list)
        return reddit_df
    except praw.exceptions.PRAWException as e:
        # Handle API error
        print("Failed to fetch Reddit posts:", str(e))
        return None


# Function to check if any of the references are mentioned in the posts


def mentioned_cryptos(posts_list, refs):
    flag = 0
    posts_string = str(posts_list)
    for ref in refs:
        if posts_string.find(ref) != -1:
            flag = 1
            break
    return flag


def extract_timestamp(title):
    if isinstance(title, str):
        # Define the regex pattern to match timestamps
        pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"

        # Search for the timestamp in the title using regex
        match = re.search(pattern, title)

        if match:
            # Extract and return the timestamp
            timestamp = match.group(1)
            return timestamp

    # If no timestamp found or title is not a string, return None or handle accordingly
    return None


# Extracting Crypto Coins Posts Related to each coin
def extract_crypto_mentions(reddit_df):
    reddit_df.columns = ["Titles", "Scores", "Timestamp"]
    reddit_df["Bitcoin"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, bitcoin_refs)
    )
    reddit_df["Ethereum"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, ether_refs)
    )
    reddit_df["XRP"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, xrp_refs)
    )
    reddit_df["Binance"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, binance_refs)
    )
    reddit_df["Litecoin"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, litecoin_refs)
    )
    reddit_df["cardano"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(
            x,
            cardano_refs,
        )
    )
    reddit_df["Dogcoin"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, dogecoin_refs)
    )
    reddit_df["Polkadot"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, polkadot_refs)
    )
    reddit_df["Stellar"] = reddit_df["Titles"].apply(
        lambda x: mentioned_cryptos(x, stellar_refs)
    )

    return reddit_df


# Applying sentimental analysis code to extracted data


def perform_sentiment_analysis(reddit_df):
    coin_mentions = {
        "BTC": reddit_df[reddit_df["Bitcoin"] == 1],
        "BINANCE": reddit_df[reddit_df["Binance"] == 1],
        "ETH": reddit_df[reddit_df["Ethereum"] == 1],
        "XRP": reddit_df[reddit_df["XRP"] == 1],
        "STL": reddit_df[reddit_df["Stellar"] == 1],
        "POL": reddit_df[reddit_df["Polkadot"] == 1],
        "DOG": reddit_df[reddit_df["Dogcoin"] == 1],
        "CRD": reddit_df[reddit_df["cardano"] == 1],
        "LTH": reddit_df[reddit_df["Litecoin"] == 1],
    }
    reddit_df["Timestamp"] = reddit_df["Titles"].apply(extract_timestamp)
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for coin, mentions in coin_mentions.items():
        for post, timestamp in zip(mentions["Titles"], mentions["Timestamp"]):
            sentiment = analyzer.polarity_scores(post)
            result = {
                "Post": post,
                "Coins": coin,
                "Timestamp": timestamp,
                "neg": sentiment["neg"],
                "neu": sentiment["neu"],
                "pos": sentiment["pos"],
                "compound": sentiment["compound"],
            }
            results.append(result)

    reddit_dfs = pd.DataFrame(results)
    return reddit_dfs


def pie_chart(reddit_df):
    btc_count = reddit_df["Coins"].value_counts()["BTC"]
    xrp_count = reddit_df["Coins"].value_counts()["XRP"]
    eth_count = reddit_df["Coins"].value_counts()["ETH"]
    bin_count = reddit_df["Coins"].value_counts()["BINANCE"]
    lin_count = reddit_df["Coins"].value_counts()["LTH"]
    # crd_count = reddit_df["Coins"].value_counts()["CRD"]
    dog_count = reddit_df["Coins"].value_counts()["DOG"]
    # pol_count = reddit_df["Coins"].value_counts()["POL"]
    stl_count = reddit_df["Coins"].value_counts()["STL"]

    total_xrp_posts = xrp_count
    total_btc_posts = btc_count
    total_etc_posts = eth_count
    total_bin_posts = bin_count
    total_lin_posts = lin_count
    # total_crd_posts = crd_count
    total_dog_posts = dog_count
    # total_pol_posts = pol_count
    total_stl_posts = stl_count

    y = np.array(
        [
            total_btc_posts,
            total_bin_posts,
            total_etc_posts,
            total_xrp_posts,
            total_lin_posts,
            # total_crd_posts,
            total_dog_posts,
            # total_pol_posts,
            total_stl_posts,
        ]
    )
    crypto_coins_labels = [
        "BTC",
        "BINANCE",
        "ETH",
        "XRP",
        "LTH",
        # "CRD",
        "DOG",
        # "POL",
        "STL",
    ]
    colors = [
        "#FFD700",
        "#FF6347",
        "#40E0D0",
        "#FFA500",
        "#ff7700",
        "#00eaff",
        # "#bb00ff",
        # "#ff006f",
        "#00ffa6",
    ]
    explode = (
        0.05,
        0.005,
        0.005,
        0.05,
        0.005,
        0.005,
        0.05,
        # 0.0,
        # 0.05,
    )

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#00172B")
    patches, texts, autotexts = ax.pie(
        y,
        labels=crypto_coins_labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
    )

    # Set text and label colors to white
    for text in texts:
        text.set_color("white")
    for autotext in autotexts:
        autotext.set_color("white")

    # ax.set_title(
    #     "Distribution of Crypto Coin Mentions According To Posts",
    #     pad=50,
    #     color="white",
    # )
    ax.legend(title="Crypto Coins", loc="best", bbox_to_anchor=(1, 0.5))
    ax.axis("equal")

    return fig


# Showing List Of Rise and Fall
def fall_rise(reddit_df):
    coins_mention = {
        "Bitcoin": reddit_df.loc[reddit_df["Coins"] == "BTC"],
        "BINANCE": reddit_df.loc[reddit_df["Coins"] == "BINANCE"],
        "ETH": reddit_df.loc[reddit_df["Coins"] == "ETH"],
        "XRP": reddit_df.loc[reddit_df["Coins"] == "XRP"],
        "LTH": reddit_df.loc[reddit_df["Coins"] == "LTH"],
        "CRD": reddit_df.loc[reddit_df["Coins"] == "CRD"],
        "DOG": reddit_df.loc[reddit_df["Coins"] == "DOG"],
        "POL": reddit_df.loc[reddit_df["Coins"] == "POL"],
        "STL": reddit_df.loc[reddit_df["Coins"] == "STL"],
    }
    results = []
    analyzer = SentimentIntensityAnalyzer()
    for coin, mentions in coins_mention.items():
        total_neg_score = 0
        total_pos_score = 0
        for post in mentions["Post"]:
            sentiment = analyzer.polarity_scores(post)
            total_neg_score += sentiment["neg"]
            total_pos_score += sentiment["pos"]

        if total_neg_score > total_pos_score:
            status = "FALL"
            arrow = "⬇️"
        elif total_neg_score == 0.000 or total_pos_score == 0.000:
            status = "Neutral"
            arrow = "⬆️"
        else:
            status = "RISE"
            arrow = "⬆️"

        result = {
            "Coin": coin,
            "NegScore": total_neg_score,
            "PosScore": total_pos_score,
            "Status": status,
            "Trending": arrow,
        }
        results.append(result)

    return results


# def line_chart(reddit_df):
#     total_compound_score = reddit_df.groupby("Coins")["compound"].sum().round()
#     coins = total_compound_score.index
#     scores = total_compound_score.values
#     max_score = max(scores)
#     min_score = min(scores)

#     # Set the figure size
#     plt.figure(figsize=(6, 4))
#     plt.plot(
#         coins,
#         scores,
#         marker="o",
#         linestyle="-",
#         linewidth=1,
#         color="b",
#     )
#     ax = plt.axes()
#     plt.title("Trending Cryptocurrencies According to Social Media", fontsize=14)
#     plt.xlabel("Crypto Coins", fontsize=13)
#     plt.ylabel("Score", fontsize=13)
#     plt.ylim(min_score, max_score + 5)
#     plt.grid(True, linestyle="--", alpha=0.4)
#     ax.set_facecolor("#0000FF")
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)

#     # Set the background color to transparent
#     plt.gca().set_facecolor("none")

#     # Labels on the points by iterating over each point
#     for x, y in zip(coins, scores):
#         plt.text(x, y, str(y), ha="center", va="bottom", fontsize=10)

#     return plt


# Bar Chart Sentiment showing sentiment scores for each crypto coin


# def bar_chart(reddit_df):
#     pos_neg_neu_df = pd.DataFrame(reddit_df)
#     total_sentiment_score = pos_neg_neu_df.groupby("Coins")[["neg", "pos", "neu"]].sum()
#     crypto_coins = total_sentiment_score.index
#     positive_scores = total_sentiment_score["pos"]
#     negative_scores = total_sentiment_score["neg"]
#     bar_width = 0.28
#     r = np.arange(len(crypto_coins))
#     plt.figure(facecolor="blue")
#     fig, ax = plt.subplots()
#     ax.bar(r, negative_scores, color="red", width=bar_width, label="Negative")
#     ax.bar(
#         r,
#         positive_scores,
#         color="green",
#         width=bar_width,
#         label="Positive",
#         bottom=negative_scores,
#     )
#     ax.set_facecolor("#00172B")
#     plt.xlabel("Crypto Coins")
#     plt.ylabel("Sentiment Scores")
#     plt.title("Distribution of Sentiment Scores for Crypto Coins")
#     plt.xticks(r, crypto_coins)
#     plt.legend()
#     return plt

# def line_chart(reddit_df):
#     total_compound_score = reddit_df.groupby("Coins")["compound"].sum().round()
#     coins = total_compound_score.index
#     scores = total_compound_score.values
#     max_score = max(scores)
#     min_score = min(scores)

#     # Create a dataframe with coins, scores, and percentages
#     data = pd.DataFrame({"Coins": coins, "Scores": scores})
#     total_scores = data["Scores"].sum()
#     data["Percentage"] = (data["Scores"] / total_scores) * 100

#     # Set the figure size and background color
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.plot(coins, scores, marker="o", linewidth=0.82, color="white")
#     ax.set_title(
#         "Trending Cryptocurrencies According to Social Media",
#         fontsize=14,
#         color="white",
#     )
#     ax.set_xlabel("Crypto Coins", fontsize=13, color="white")
#     ax.set_ylabel("Score", fontsize=13, color="white")
#     ax.set_ylim(min_score, max_score + 5)
#     ax.grid(True, linestyle="--", alpha=0.4)
#     ax.set_facecolor("#00172B")
#     ax.tick_params(axis="x", labelrotation=90, labelsize=10, color="white")
#     ax.tick_params(axis="y", labelsize=10, color="white")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_color("white")
#     ax.spines["left"].set_color("white")
#     ax.xaxis.set_tick_params(width=0.5, color="white", labelcolor="white")
#     ax.yaxis.set_tick_params(width=0.5, color="white", labelcolor="white")

#     # Display labels and percentages
#     for coin, score, percentage in zip(coins, scores, data["Percentage"]):
#         ax.text(
#             coin,
#             score,
#             f"{score}\n{percentage:.2f}%",
#             ha="center",
#             va="bottom",
#             fontsize=10,
#             color="white",
#         )

#     # Draw x-axis line
#     ax.axhline(y=0, color="white", linewidth=0.5)

#     # Draw y-axis line
#     ax.axvline(x=0, color="white", linewidth=0.5)

#     # Remove the default plot axes and background
#     plt.box(False)
#     plt.axis("off")

#     # Set the background color of the figure
#     fig.patch.set_facecolor("#00172B")

#     # Display the modified plot in Streamlit
#     return fig
