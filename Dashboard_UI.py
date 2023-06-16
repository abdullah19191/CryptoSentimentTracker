import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from backend import (
    authenticate_reddit,
    fetch_reddit_posts,
    extract_crypto_mentions,
    perform_sentiment_analysis,
    pie_chart,
    fall_rise,
)

st.set_page_config(
    layout="wide",
    page_title="Crypto Sentiment Analysis Dashboard",
    page_icon=":bar_chart:",
)


def line_chart(reddit_df):
    total_compound_score = reddit_df.groupby("Coins")["compound"].sum().round()
    coins = total_compound_score.index
    scores = total_compound_score.values
    max_score = max(scores)
    min_score = min(scores)

    # Create a dataframe with coins, scores, and percentages
    data = pd.DataFrame({"Coins": coins, "Scores": scores})
    total_scores = data["Scores"].sum()
    data["Percentage"] = (data["Scores"] / total_scores) * 100

    # Set the background color
    st.markdown(
        f"""
        <style>
        .stGraph > div > div > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div > svg {{
            background-color: #00172B !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create a dictionary to map x-axis labels
    x_labels = {i: coin for i, coin in enumerate(coins)}

    # Draw the line chart with custom x-axis labels
    chart_data = pd.Series(data=scores, index=pd.Index(coins, name="Coins"))
    st.markdown(
        "Total sentiment score for each cryptocurrency,visualizing them using a line Chart"
    )
    st.line_chart(
        chart_data, use_container_width=True, x=x_labels, height=480, width=100
    )


def bar_chart(reddit_df):
    pos_neg_neu_df = pd.DataFrame(reddit_df)
    total_sentiment_score = pos_neg_neu_df.groupby("Coins")[["neg", "pos", "neu"]].sum()
    crypto_coins = total_sentiment_score.index
    positive_scores = total_sentiment_score["pos"]
    negative_scores = total_sentiment_score["neg"]

    # Create a dataframe with crypto coins and sentiment scores
    data = pd.DataFrame(
        {
            "Negative": negative_scores,
            "Positive": positive_scores,
        }
    )

    # Set the bar color for negative and positive
    colors = ["#0083B8", "#004db8"]

    # Create a Streamlit bar chart
    st.markdown("  Bar Chart Sentiment showing sentiment scores for each crypto coin")
    st.bar_chart(data, use_container_width=True, height=500, width=100)


def main():
    st.title(":bar_chart:CryptoSentimentTracker")
    st.markdown("##")
    # Authenticate with Reddit API
    reddit_client = authenticate_reddit()
    if reddit_client is None:
        st.error(
            "Failed to authenticate with Reddit API. Please check your credentials."
        )
        return

    st.write("Authenticated:", reddit_client.read_only)

    # Fetch Reddit posts
    num_posts = 550
    reddit_posts = fetch_reddit_posts(reddit_client, num_posts)
    if reddit_posts is None:
        st.error("Failed to fetch Reddit posts. Please try again later.")
        return

    crypto_mentions = extract_crypto_mentions(reddit_posts)
    sentiment_analysis = perform_sentiment_analysis(crypto_mentions)
    # st.dataframe(sentiment_analysis)
    total_posts = num_posts
    total_cryptos = 12
    star_rating = ":star:" * 4

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Posts")
        st.subheader(f" Posts:   {total_posts:,}")
    with middle_column:
        st.subheader("Total Crypto Currency")
        st.subheader(f" {total_cryptos} {star_rating}")
    with right_column:
        st.subheader("Total Compound Score")
        st.subheader(f"Profit Percentage: 64% ")

    st.markdown("""---""")

    # st.dataframe(sentiment_analysis)
    st.sidebar.image("logoS2.png", use_column_width=False, width=260)
    st.sidebar.header("Please Filter Here:")
    sentiment_options = ["Fall", "Neutral", "Rise"]
    selected_sentiment = st.sidebar.selectbox(
        "Select Sentiment Status", sentiment_options
    )
    sentiment_options2 = [
        "BTC",
        "BINANCE",
        "ETH",
        "XRP",
        "LTH",
        "DOG",
        "CRD",
        "POL",
        "STL",
    ]
    selected_sentiment2 = st.sidebar.selectbox(
        "Select Crypto Coins", sentiment_options2
    )
    st.sidebar.slider(
        "Number of Posts",
        min_value=100,
        max_value=850,
        value=550,
        step=10,
    )
    col1, col2 = st.columns(2)
    with col1:
        # Create the line chart card
        line_chart(sentiment_analysis)
        # Create the rise and fall list card
        st.markdown("Trending Crypto Coins")
        list_fall_rise = fall_rise(sentiment_analysis)
        st.dataframe(list_fall_rise, width=1080)
        list_rise = fall_rise(sentiment_analysis)
        filtered_results = [
            result for result in list_rise if result["Status"] == selected_sentiment
        ]
        # st.header("Filtered Results")
        # st.dataframe(filtered_results)
    st.markdown("""---""")
    st.header("Unveiling Crypto Sentiment, Empowering Investors")
    # Second column
    with col2:
        with st.container():
            bar_chart(sentiment_analysis)
            # Create the pie chart card
            sentiment_pie_chart = pie_chart(sentiment_analysis)
            st.markdown("Distribution of Crypto Coin Mentions According To Posts")
            st.pyplot(sentiment_pie_chart)
        # Create the bar chart card

    # sentiment_line_chart = line_chart(sentiment_analysis)
    # st.pyplot(sentiment_line_chart)


if __name__ == "__main__":
    main()


# coins = st.sidebar.multiselect(
#     "Select Crypto Coins: ",
#     options=reddit_df["first column"].unique(),
#     default=df["first column"].unique(),
# )
