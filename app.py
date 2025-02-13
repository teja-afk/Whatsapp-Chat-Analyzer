import streamlit as st
import pandas as pd
import nltk
from io import StringIO
from processes import (
    analyze_sentiment, preprocess, filter_by_date, fetch_stats,
    monthly_timeline, daily_timeline, Week_activity, Monthly_activity,
    chat_contri, most_common_words, emoji, create_wordcloud, activity_heatmap,
    response_time_analysis, detect_spam_and_important, generate_pdf
)

# Download required nltk datasets
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Sidebar Title
st.sidebar.title("WhatsApp Chat Analyzer")

# File Upload
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    if "file_data" not in st.session_state:
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        df = preprocess(string_data)
        st.session_state.file_data = df  # Store processed data
    else:
        df = st.session_state.file_data  # Use cached data

    st.dataframe(df, use_container_width=True)

    if "user" in df.columns:
        user_list = df["user"].unique().tolist()
        user_list = list(filter(lambda x: x != "group_notification", user_list))
        user_list.insert(0, "Overall")

        st.sidebar.subheader("Filters")
        date_filter = st.sidebar.date_input("Select a date range", [])
        if date_filter and len(date_filter) == 2:
            df = filter_by_date(df, *date_filter)

        selected_user = st.sidebar.selectbox("Filter by User:", user_list, key="user_filter")
        if selected_user != "Overall":
            df = df[df["user"] == selected_user]

    selected = st.sidebar.selectbox("Show analysis with respect to:", user_list, key="analysis_filter")

    if st.sidebar.button("Show analysis", key="show_analysis_button"):
        st.session_state.analysis_selected = selected

        # Fetch Stats
        no_messages, no_words, no_media, no_urls = fetch_stats(selected, df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", no_messages)
        col2.metric("Total Words", no_words)
        col3.metric("Total Media", no_media)
        col4.metric("Total Links", no_urls)

        # Monthly Timeline
        st.subheader("Monthly Timeline")
        monthly_timeline_data = monthly_timeline(selected, df)
        st.line_chart(monthly_timeline_data, x=monthly_timeline_data.columns[1], y=monthly_timeline_data.columns[0])

        # Daily Timeline
        st.subheader("Daily Timeline")
        daily_timeline_data = daily_timeline(selected, df)
        st.pyplot(daily_timeline_data)

        # Activity Analysis
        col1, col2 = st.columns(2)
        col1.subheader("Most Busy Days (Weekly)")
        col1.pyplot(Week_activity(selected, df))
        col2.subheader("Most Busy Days (Monthly)")
        col2.pyplot(Monthly_activity(selected, df))

        # Chat Contribution (Fixing Overlapping Usernames in Bar Chart)
        if selected == "Overall":
            st.subheader("Chat Contribution")
            chat_chart, df_contri = chat_contri(df)
            st.pyplot(chat_chart)  # Fixed overlapping issue in `chat_contri()`
            st.dataframe(df_contri, use_container_width=True)

            # Download Options
            summary_df = pd.DataFrame({"Metric": ["Total Messages", "Total Words", "Total Media", "Total Links"],
                                       "Value": [no_messages, no_words, no_media, no_urls]})
            csv_summary = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Chat Summary as CSV", csv_summary, "chat_summary.csv", "text/csv")

            csv_contri = df_contri.to_csv(index=False).encode("utf-8")
            st.download_button("Download Chat Contribution as CSV", csv_contri, "chat_contribution.csv", "text/csv")

            detailed_df = df[["dates", "user", "messages"]]
            csv_detailed = detailed_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Detailed Messages as CSV", csv_detailed, "detailed_messages.csv", "text/csv")

        # Most Common Words
        st.subheader("Most Common Words")
        st.pyplot(most_common_words(selected, df))

        # Emoji Analysis
        col1, col2 = st.columns(2)
        col1.subheader("Most Common Emojis")
        col1.dataframe(emoji(selected, df), use_container_width=True)

        # Word Cloud
        st.subheader("Word Cloud")
        st.pyplot(create_wordcloud(selected, df))

        # Weekly Activity Heatmap
        st.subheader("Weekly Activity Map")
        st.pyplot(activity_heatmap(selected, df))

        # Response Time Analysis
        st.subheader("Average Response Time")
        avg_response_time = response_time_analysis(df)
        st.dataframe(avg_response_time, use_container_width=True)

        # Spam & Important Message Analysis
        st.subheader("Message Analysis")
        spam_count, non_spam_count, spam_msgs, important_count, important_msgs = detect_spam_and_important(df, selected)

        col1, col2, col3 = st.columns(3)
        col1.metric("Spam Messages", spam_count)
        col2.metric("Non-Spam Messages", non_spam_count)
        col3.metric("Important Messages", important_count)

        # Display Messages
        st.subheader("Spam Messages")
        st.dataframe(spam_msgs[["dates", "user", "messages"]], use_container_width=True)

        st.subheader("Important Messages")
        st.dataframe(important_msgs[["dates", "user", "messages"]], use_container_width=True)

        # PDF Report Generation
        pdf_output = generate_pdf(df, {"spam_count": spam_count, "important_count": important_count,
                                       "neutral_count": non_spam_count, "spam_messages": spam_msgs,
                                       "important_messages": important_msgs})
        st.download_button("Download Analysis Report as PDF", pdf_output, "chat_analysis_report.pdf", "application/pdf")

        # Restore selected user in dropdown after refresh
        if "analysis_selected" in st.session_state:
            st.sidebar.selectbox("Show analysis with respect to:", user_list,
                                 index=user_list.index(st.session_state.analysis_selected), key="restored_analysis_filter")

        # Sentiment Analysis
        if "messages" in df.columns:
            df["Polarity"], df["Sentiment"] = zip(*df["messages"].apply(analyze_sentiment))
        
        st.subheader("Sentiment Analysis")
        if "Sentiment" in df.columns and not df["Sentiment"].isna().all():
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
            st.write("Sentiment Counts:", sentiment_counts)
        else:
            st.warning("No sentiment data available.")

################################### CSS ###########################################3333
