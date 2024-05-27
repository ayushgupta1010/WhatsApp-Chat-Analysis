import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import helper
import preprocessor

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Sentiment Analysis
        st.title("Sentiment Analysis")
        sentiment = helper.sentiment_analysis(selected_user, df)
        st.write(
            f"Overall Sentiment: Positive - {sentiment['positive']:.2f}, Negative - {sentiment['negative']:.2f}, Neutral - {sentiment['neutral']:.2f}")

        # Pie chart for sentiment analysis
        fig, ax_sentiment = plt.subplots()
        ax_sentiment.pie(sentiment.values(), labels=sentiment.keys(), autopct='%1.1f%%', startangle=140)
        ax_sentiment.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc is not None:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
        else:
            st.write("Not enough words available to generate a word cloud.")

        # most common words
        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        if most_common_df is not None and not most_common_df.empty:
            # Convert the Word column to strings to ensure compatibility with plotting
            most_common_df['Word'] = most_common_df['Word'].astype(str)

            fig, ax = plt.subplots()
            ax.barh(most_common_df['Word'], most_common_df['Frequency'])
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            plt.title('Most Common Words')
            plt.gca().invert_yaxis()  # Invert y-axis to display most common words at the top
            plt.tight_layout()  # Adjust layout to prevent clipping of labels
            st.pyplot(fig)
        else:
            st.write("No words available for most common words analysis.")


        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        if not emoji_df.empty:
            st.dataframe(emoji_df)
        else:
            st.write("No emoji data available for selected user.")
