# import streamlit as st
# from io import StringIO
# import pandas as pd
# import processes

# st.sidebar.title("Whatsapp Chat Analyzer")
# uploaded_file = st.sidebar.file_uploader("Choose a file")

# if uploaded_file is not None:
#     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#     string_data = stringio.read()

#     df = processes.preprocess(string_data)
#     st.dataframe(df, use_container_width=True)

#     if 'user' in df.columns:
#         user_list = df['user'].unique().tolist()
#         user_list = list(filter(lambda x:x != 'group_notification',user_list))
#         unique_users = user_list
#         user_list.insert(0,"Overall")
#         ## Customizable filters #####################
#         st.sidebar.subheader("Filters")

#         # Date filter
#         date_filter = st.sidebar.date_input("Select a date range", [])
#         if date_filter and len(date_filter) == 2:
#             start_date, end_date = date_filter
#             df = processes.filter_by_date(df, start_date, end_date)

#         # User filter
#         selected_user = st.sidebar.selectbox("Filter by User:", user_list)
#         if selected_user != "Overall":
#             df = df[df['user'] == selected_user]

#     selected = st.sidebar.selectbox("show analysis with respect to:",user_list)
#     if st.sidebar.button("Show analysis"):
#         no_messages,no_words,no_media, no_urls = processes.fetch_stats(selected,df)

#         col1, col2, col3,col4 = st.columns(4)

#         with col1:
#             st.header("Total Messages")
#             st.title(no_messages)

#         with col2:
#             st.header("Total Words")
#             st.title(no_words)

#         with col3:
#             st.header("Total Media")
#             st.title(no_media)

#         with col4:
#             st.header("Total Links")
#             st.title(no_urls)

#         #Monthly timeline
#         st.title("Monthly timeline:")
#         monthly_timeline = processes.monthly_timeline(selected,df)
#         st.line_chart(monthly_timeline, x=monthly_timeline.columns[1],y= monthly_timeline.columns[0])

#         #Daily timeline
#         st.title("Daily timeline:")
#         daily_timeline = processes.daily_timeline(selected,df)
#         st.pyplot(daily_timeline)

#         col1, col2 = st.columns(2)

#         with col1:
#             #Week Activity
#             st.title("Most busy days Weekly:")
#             week_activity = processes.Week_activity(selected,df)
#             st.pyplot(week_activity)

#         with col2:
#             #Monthly Activity
#             st.title("Most busy days Monthly:")
#             monthly_activity = processes.Monthly_activity(selected,df)
#             st.pyplot(monthly_activity)


# #################################### export data button added################################################### 
#         if selected == 'Overall':
#             st.title("Chat Contribution:")
#             chart, df_contri = processes.chat_contri(df)
#             st.pyplot(chart)
#             st.dataframe(df_contri, use_container_width=True)

#             # Calculate other statistics
#             no_messages, no_words, no_media, no_urls = processes.fetch_stats(selected, df)

#             # Create a summary DataFrame for overall statistics
#             summary_data = {
#                 "Metric": ["Total Messages", "Total Words", "Total Media", "Total Links"],
#                 "Value": [no_messages, no_words, no_media, no_urls]
#             }

#             # Debugging: Check lengths
#             for key, value in summary_data.items():
#                 st.write(f"{key}: {len(value)}")

#             try:
#                 summary_df = pd.DataFrame(summary_data)
#                 st.write("Summary DataFrame created successfully.")
#             except ValueError as e:
#                 st.error(f"Error creating DataFrame: {e}")

#             # Data export functionality for overall stats
#             csv_summary = summary_df.to_csv(index=False).encode('utf-8')  # Convert summary DataFrame to CSV
#             st.download_button(
#                 label="Download Overall Chat Analysis as CSV",
#                 data=csv_summary,
#                 file_name='whatsapp_chat_analysis_summary.csv',
#                 mime='text/csv'
#             )

#             # Data export functionality for chat contributions
#             csv_contri = df_contri.to_csv(index=False).encode('utf-8')  # Convert contribution DataFrame to CSV
#             st.download_button(
#                 label="Download Chat Contribution Analysis as CSV",
#                 data=csv_contri,
#                 file_name='whatsapp_chat_analysis_contribution.csv',
#                 mime='text/csv'
#             )

#             # New: Data export functionality for detailed user messages
#             detailed_df = df[['dates', 'user', 'messages']]  # Select relevant columns
#             csv_detailed = detailed_df.to_csv(index=False).encode('utf-8')  # Convert detailed DataFrame to CSV
#             st.download_button(
#                 label="Download Detailed User Messages as CSV",
#                 data=csv_detailed,
#                 file_name='whatsapp_chat_analysis_detailed_messages.csv',
#                 mime='text/csv'
#             )

# ##################################################################################################################

#         st.title("Most common words:")
#         most_common_graph = processes.most_common_words(selected,df)
#         st.pyplot(most_common_graph)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.title("Most common Emojis:")
#             emoji_df = processes.emoji(selected,df)
#             st.dataframe(emoji_df,use_container_width=True)
            
#         st.title("Word Cloud:")
#         cloud_img = processes.create_wordcloud(selected,df)
#         st.pyplot(cloud_img)

#         st.title("Weekly Activity Map")
#         activity_heatmap = processes.activity_heatmap(selected,df)
#         st.pyplot(activity_heatmap)
       
#         #response time analysis 
#         st.title("Average Response Time:")
#         avg_response_time = processes.response_time_analysis(df)
#         st.dataframe(avg_response_time, use_container_width=True)

#         # Spam and important messages detection
#         st.title("Message Analysis:")
#         spam_count, non_spam_count, spam_messages, important_count, important_messages = processes.detect_spam_and_important(df, selected)

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.header("Spam Messages Count")
#             st.title(spam_count)

#         with col2:
#             st.header("Non-Spam Messages Count")
#             st.title(non_spam_count)

#         with col3:
#             st.header("Important Messages Count")
#             st.title(important_count)

#         st.title("Spam Messages:")
#         st.dataframe(spam_messages[['dates', 'user', 'messages']], use_container_width=True)

#         st.title("Important Messages:")
#         st.dataframe(important_messages[['dates', 'user', 'messages']], use_container_width=True)
import streamlit as st
from io import StringIO
import pandas as pd
import processes

st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    # Store file contents in session state to persist data across reruns
    if "file_data" not in st.session_state:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        df = processes.preprocess(string_data)
        st.session_state.file_data = df  # Save preprocessed data
    else:
        df = st.session_state.file_data  # Use saved data if available

    st.dataframe(df, use_container_width=True)

    if 'user' in df.columns:
        user_list = df['user'].unique().tolist()
        user_list = list(filter(lambda x: x != 'group_notification', user_list))
        unique_users = user_list
        user_list.insert(0, "Overall")

        # Customizable filters #####################
        st.sidebar.subheader("Filters")

        # Date filter
        date_filter = st.sidebar.date_input("Select a date range", [])
        if date_filter and len(date_filter) == 2:
            start_date, end_date = date_filter
            df = processes.filter_by_date(df, start_date, end_date)

        # User filter
        selected_user = st.sidebar.selectbox("Filter by User:", user_list, key="user_filter")
        if selected_user != "Overall":
            df = df[df['user'] == selected_user]

    # Make sure the selectbox gets a unique key in each call
    selected = st.sidebar.selectbox("Show analysis with respect to:", user_list, key="analysis_filter")
    if st.sidebar.button("Show analysis", key="show_analysis_button"):
        # Store analysis state in session to avoid losing it
        st.session_state.analysis_selected = selected
        no_messages, no_words, no_media, no_urls = processes.fetch_stats(selected, df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(no_messages)

        with col2:
            st.header("Total Words")
            st.title(no_words)

        with col3:
            st.header("Total Media")
            st.title(no_media)

        with col4:
            st.header("Total Links")
            st.title(no_urls)

        # Monthly timeline
        st.title("Monthly timeline:")
        monthly_timeline = processes.monthly_timeline(selected, df)
        st.line_chart(monthly_timeline, x=monthly_timeline.columns[1], y=monthly_timeline.columns[0])

        # Daily timeline
        st.title("Daily timeline:")
        daily_timeline = processes.daily_timeline(selected, df)
        st.pyplot(daily_timeline)

        col1, col2 = st.columns(2)

        with col1:
            # Week Activity
            st.title("Most busy days Weekly:")
            week_activity = processes.Week_activity(selected, df)
            st.pyplot(week_activity)

        with col2:
            # Monthly Activity
            st.title("Most busy days Monthly:")
            monthly_activity = processes.Monthly_activity(selected, df)
            st.pyplot(monthly_activity)

        if selected == 'Overall':
            st.title("Chat Contribution:")
            chart, df_contri = processes.chat_contri(df)
            st.pyplot(chart)
            st.dataframe(df_contri, use_container_width=True)

            summary_data = {
                "Metric": ["Total Messages", "Total Words", "Total Media", "Total Links"],
                "Value": [no_messages, no_words, no_media, no_urls]
            }

            try:
                summary_df = pd.DataFrame(summary_data)
                st.write("Summary DataFrame created successfully.")
            except ValueError as e:
                st.error(f"Error creating DataFrame: {e}")

            # Data export functionality for overall stats
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Overall Chat Analysis as CSV",
                data=csv_summary,
                file_name='whatsapp_chat_analysis_summary.csv',
                mime='text/csv'
            )

            csv_contri = df_contri.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Chat Contribution Analysis as CSV",
                data=csv_contri,
                file_name='whatsapp_chat_analysis_contribution.csv',
                mime='text/csv'
            )

            detailed_df = df[['dates', 'user', 'messages']]
            csv_detailed = detailed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed User Messages as CSV",
                data=csv_detailed,
                file_name='whatsapp_chat_analysis_detailed_messages.csv',
                mime='text/csv'
            )

        st.title("Most common words:")
        most_common_graph = processes.most_common_words(selected, df)
        st.pyplot(most_common_graph)

        col1, col2 = st.columns(2)

        with col1:
            st.title("Most common Emojis:")
            emoji_df = processes.emoji(selected, df)
            st.dataframe(emoji_df, use_container_width=True)

        st.title("Word Cloud:")
        cloud_img = processes.create_wordcloud(selected, df)
        st.pyplot(cloud_img)

        st.title("Weekly Activity Map")
        activity_heatmap = processes.activity_heatmap(selected, df)
        st.pyplot(activity_heatmap)

        # Response time analysis
        st.title("Average Response Time:")
        avg_response_time = processes.response_time_analysis(df)
        st.dataframe(avg_response_time, use_container_width=True)

        # Spam and important messages detection
        st.title("Message Analysis:")
        spam_count, non_spam_count, spam_messages, important_count, important_messages = processes.detect_spam_and_important(df, selected)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Spam Messages Count")
            st.title(spam_count)

        with col2:
            st.header("Non-Spam Messages Count")
            st.title(non_spam_count)

        with col3:
            st.header("Important Messages Count")
            st.title(important_count)

        st.title("Spam Messages:")
        st.dataframe(spam_messages[['dates', 'user', 'messages']], use_container_width=True)

        st.title("Important Messages:")
        st.dataframe(important_messages[['dates', 'user', 'messages']], use_container_width=True)

        # Generate PDF Report
        pdf_output = processes.generate_pdf(df, {
            "spam_count": spam_count,
            "important_count": important_count,
            "neutral_count": non_spam_count,
            "spam_messages": spam_messages,
            "important_messages": important_messages,
        })  # Generate PDF

        # PDF Download Button
        st.download_button(
            label="Download Analysis Report as PDF",
            data=pdf_output,
            file_name="chat_analysis_report.pdf",
            mime="application/pdf"
        )

    # Other download options (CSV, Excel)...
        
        
        # Restore analysis on rerun
        if 'analysis_selected' in st.session_state:
            st.sidebar.selectbox("Show analysis with respect to:", user_list, index=user_list.index(st.session_state.analysis_selected), key="restored_analysis_filter")

        