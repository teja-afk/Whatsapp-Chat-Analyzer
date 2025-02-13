import os
import re
import io
import textwrap
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
from textblob import TextBlob

nltk.download('vader_lexicon')

def preprocess(data):

    pattern = '\d{1,2}\/\d{1,2}\/\d{1,2},\s\d{1,2}:\d{2}\s-\s|\[\d{1,2}\/\d{1,2}\/\d{1,2}, \d{1,2}:\d{1,2}:\d{1,2}.[AaPp][Mm]\]'
    messages = re.split(pattern,data)[1:]
    dates = re.findall(pattern,data)
    dates = [ d.strip("[]") for d in dates]

    #Create Dataframe from messages and dates
    df = pd.DataFrame({'messages':messages,'dates':dates})
    date_string = dates[0]
    format_string = '%d/%m/%y, %I:%M:%S %p'

    #Convert dates to datetime
    try:
        res = bool(datetime.strptime(date_string, format_string))
    except Exception:
        res = False

    if res == True:
        df['dates'] = pd.to_datetime(df['dates'],format= '%d/%m/%y, %I:%M:%S %p')
    else:
        df['dates'] = pd.to_datetime(df['dates'],format= '%d/%m/%y, %H:%M - ')

    # Intialize user and message lists
    user = []
    message = []

    for m in messages:
        entry  = m.split(": ")
        if len(entry) == 2:
            user.append(entry[0])
            message.append(entry[1])
        else:
            user.append('group_notification')
            message.append(entry[0])

    df['user'] = user
    df['messages'] = message

    # Extract additional date features    
    df['year'] = df['dates'].dt.year
    df['months'] = df['dates'].dt.month_name()
    df['day'] = df['dates'].dt.day
    df['hour'] = df['dates'].dt.hour
    df['minute'] = df['dates'].dt.minute
    df['month_num'] = df['dates'].dt.month
    df['day_name'] = df['dates'].dt.day_name()
    df['date'] = df['dates'].dt.date

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df

def fetch_stats(selected,df):

    if selected != 'Overall':
        df = df[df['user'] == selected]

    no_messages = df.shape[0]

    words = []
    for mes in df['messages']:
        words.extend(mes.split(" "))
    no_words = len(words)

    no_media = df[df['messages'] == '<Media omitted>\n'].shape[0]

    extractor = URLExtract()
    urls = []

    for mes in df['messages']:
        url = extractor.find_urls(mes)
        urls.extend(url)

    no_urls = len(urls)

    return no_messages,no_words,no_media, no_urls

def chat_contri(df):
    # Count messages per user
    user_counts = df['user'].value_counts()

    # Set figure size
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bar chart
    user_counts.plot(kind='bar', ax=ax, color='skyblue')

    # Set labels and title
    ax.set_xlabel("User")
    ax.set_ylabel("Message Count")
    ax.set_title("Chat Contribution")

    # Rotate x-axis labels and wrap long names
    wrapped_labels = [textwrap.fill(name, width=10) for name in user_counts.index]
    ax.set_xticklabels(wrapped_labels, rotation=30, ha="right", fontsize=9)

    plt.xticks(rotation=30, ha='right')  # Ensures labels are properly aligned
    plt.tight_layout()  # Adjust layout to fit labels properly

    return fig

def most_common_words(selected,df):
    f = open('stop_hinglish.txt','r')
    stopwords = f.read()
    f.close()
    if selected != 'Overall':
        df = df[df['user'] == selected]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['messages'] != '<Media omitted>\n']
    word_list = []
    for mes in temp['messages']:
        for word in mes.lower().split():
            if word not in stopwords:
                word_list.append(word)
    most_common_df = pd.DataFrame(Counter(word_list).most_common(20))
    fig3,axis = plt.subplots()
    axis.barh(most_common_df[0],most_common_df[1])
    plt.xticks(rotation= 'vertical')

    return fig3

def emoji(selected,df):
    import emoji
    if selected != 'Overall':
        df = df[df['user'] == selected]
    emojis = []
    for m in df['messages']:
        for c in m:
            if c in emoji.EMOJI_DATA:
                emojis.append(c)
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected,df):
    if selected != 'Overall':
        df = df[df['user'] == selected]
    monthly_timeline = df.groupby(['year','month_num','months']).count()['messages'].reset_index()
    time = []
    for i in range(monthly_timeline.shape[0]):
        time.append(monthly_timeline['months'][i] + "-" + str(monthly_timeline['year'][i]))
    monthly_timeline['time'] = time
    monthly_timeline = monthly_timeline[['messages','time']]
    return monthly_timeline

def daily_timeline(selected,df):

    if selected != 'Overall':
        df = df[df['user'] == selected]

    daily_timeline = df.groupby('date').count()['messages'].reset_index()

    fig4,axis = plt.subplots()
    axis.plot(daily_timeline['date'],daily_timeline['messages'], color='green')
    plt.xticks(rotation='vertical')

    return fig4

def Week_activity(selected,df):

    if selected != 'Overall':
        df = df[df['user'] == selected]  

    week_activity = df['dates'].dt.day_name().value_counts().reset_index()

    fig5,axis = plt.subplots()
    axis.bar(week_activity['dates'],week_activity['count'])
    plt.xticks(rotation= 'vertical')
    return fig5 

def Monthly_activity(selected,df):

    if selected != 'Overall':
        df = df[df['user'] == selected]  

    monthly_activity = df['months'].value_counts().reset_index()

    fig6,axis = plt.subplots()
    axis.bar(monthly_activity['months'],monthly_activity['count'],color='red')
    plt.xticks(rotation= 'vertical')
    return fig6 

def create_wordcloud(selected,df):

    f = open('stop_hinglish.txt','r')
    stopwords = f.read()
    f.close()

    if selected != 'Overall':
        df = df[df['user'] == selected]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['messages'] != '<Media omitted>\n']

    def remove_stopwords(message):
        y = []
        for word in message.lower().split():
            if word not in stopwords:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width = 800, height = 800,background_color ='white',
                    min_font_size = 10)
    temp['messages'].apply(remove_stopwords)
    wc_df = wc.generate(temp['messages'].str.cat(sep=" "))

    fig2,axis = plt.subplots()
    axis.imshow(wc_df)
    return fig2

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='messages', aggfunc='count').fillna(0)
    fig7,axis = plt.subplots()
    axis = sns.heatmap(user_heatmap)

    return fig7

def response_time_analysis(df):
    # Convert 'dates' to datetime if not already
    df['dates'] = pd.to_datetime(df['dates'])
    
    # Sort by user and date
    df = df.sort_values(by=['user', 'dates'])
    
    # Calculate response times
    df['response_time'] = df.groupby('user')['dates'].diff().dt.total_seconds() / 60  # in minutes
    
    # Calculate average response time per user
    avg_response_time = df.groupby('user')['response_time'].mean().reset_index()
    avg_response_time.columns = ['user', 'avg_response_time (in minutes)']
    
    return avg_response_time

def detect_spam_and_important(df, selected=None, spam_keywords=None, important_keywords=None, emoji_threshold=5, repetition_threshold=3):
    if spam_keywords is None:
        spam_keywords = [
            r'\bbuy\s+now\b', r'\bdiscount\b', r'\bfree\b', r'\bearn\s+money\b',
            r'\blimited\s+time\b', r'\bclick\s+here\b', 
            r'\bcongratulations\b', r'\bwon\b'
        ]
    
    if important_keywords is None:
        important_keywords = [
            r'\bmeeting\b', r'\bdeadline\b', r'\bimportant\b', r'\btest\b'
            r'\burgent\b', r'\baction required\b', r'\bplease respond\b', r'\bfees\b', r'\bjoin immediately\b', r'\bsubmission date\b', r'\bnotice\b', r'\bcome in lab\b',
            r'\btomorrow\b', r'call me immediately', r'\bmeet me immediately\b', r'\bwithout fail\b'
        ]

    if selected != 'Overall' and selected is not None:
        df = df[df['user'] == selected]

    def is_spam(message):
        # Check for spam keywords
        if any(re.search(pattern, message.lower()) for pattern in spam_keywords):
            return True

        # Check for excessive emojis/symbols
        emoji_count = len(re.findall(r'[^\w\s]', message))
        if emoji_count > emoji_threshold:
            return True

        return False

    def is_important(message):
        # Check for important keywords
        if any(re.search(pattern, message.lower()) for pattern in important_keywords):
            return True
        return False

    # Apply spam and important checks
    df['is_spam'] = df['messages'].apply(is_spam)
    df['is_important'] = df['messages'].apply(is_important)
    
    # Detect repeated messages for spam
    message_counts = df['messages'].value_counts()
    repeated_messages = message_counts[message_counts > repetition_threshold].index
    df['is_spam'] = df.apply(lambda row: row['messages'] in repeated_messages or row['is_spam'], axis=1)

    # Count spam and important messages
    spam_count = df['is_spam'].sum()
    non_spam_count = df.shape[0] - spam_count
    important_count = df['is_important'].sum()

    # Get spam and important messages
    spam_messages = df[df['is_spam']].reset_index(drop=True)
    important_messages = df[df['is_important']].reset_index(drop=True)

    return spam_count, non_spam_count, spam_messages, important_count, important_messages

def chat_contri(filtered_df, top_n=10):
    """
    Generates a bar chart showing the top contributors in the chat.

    Parameters:
    - filtered_df: DataFrame containing chat data with a 'user' column.
    - top_n: Number of top users to display (default is 10).

    Returns:
    - fig: Matplotlib figure object.
    - df_contri: DataFrame with contribution data.
    """

    # Count messages per user
    df_contri = filtered_df['user'].value_counts(normalize=True).mul(100).reset_index()
    df_contri.columns = ['user', 'percent']
    
    # Select top contributors
    df_top = df_contri.head(top_n)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use seaborn for better aesthetics
    sns.barplot(x=df_top['user'], y=df_top['percent'], palette='viridis', ax=ax)

    # Labels and formatting
    ax.set_title('Top Chat Contributors', fontsize=14)
    ax.set_xlabel('User', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10)  # Rotate for readability
    plt.yticks(fontsize=10)

    # Display percentage values on bars
    for index, value in enumerate(df_top['percent']):
        ax.text(index, value + 1, f"{value:.1f}%", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()  # Adjust layout to prevent cutoff

    return fig, df_contri

def filter_by_date(df, start_date, end_date):
    # Ensure start_date and end_date are converted to the correct type
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['dates'] >= start_date) & (df['dates'] <= end_date)]
    return filtered_df

def clean_text(text):
    # Replace or remove unsupported characters
    return re.sub(r'[^\x20-\x7E]', '', text)  # Removes non-ASCII characters

def generate_pdf(df, analysis_data):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="WhatsApp Chat Analysis Report", ln=True, align='C')

    # Total stats
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Analysis Summary", ln=True)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"Total Messages: {analysis_data['spam_count'] + analysis_data['important_count'] + analysis_data['neutral_count']}", ln=True)
    pdf.cell(200, 10, txt=f"Spam Messages: {analysis_data['spam_count']}", ln=True)
    pdf.cell(200, 10, txt=f"Important Messages: {analysis_data['important_count']}", ln=True)
    pdf.cell(200, 10, txt=f"Neutral Messages: {analysis_data['neutral_count']}", ln=True)

    # Add message breakdown tables
    pdf.ln(10)  # Line break
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Detailed Breakdown", ln=True)

    # Add spam messages (optional)
    if not analysis_data["spam_messages"].empty:
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt="Spam Messages", ln=True)
        pdf.set_font('Arial', '', 10)
        for idx, row in analysis_data["spam_messages"].head(10).iterrows():
            clean_message = clean_text(f"{row['dates']} - {row['user']}: {row['messages']}")
            pdf.cell(200, 8, txt=clean_message, ln=True)

    # Add important messages (optional)
    if not analysis_data["important_messages"].empty:
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt="Important Messages", ln=True)
        pdf.set_font('Arial', '', 10)
        for idx, row in analysis_data["important_messages"].head(10).iterrows():
            clean_message = clean_text(f"{row['dates']} - {row['user']}: {row['messages']}")
            pdf.cell(200, 8, txt=clean_message, ln=True)

    # Save the PDF to a temporary file
    temp_file_path = 'chat_analysis_report.pdf'
    pdf.output(temp_file_path)

    # Read the file as bytes
    with open(temp_file_path, 'rb') as f:
        pdf_output = f.read()

    # Optionally, remove the temporary file after reading it
    os.remove(temp_file_path)

    return pdf_output

def analyze_sentiment(message):
    """
    Analyzes the sentiment of a message.
    Returns:
        polarity (float): The sentiment polarity (-1 to 1).
        sentiment (str): 'Positive', 'Negative', or 'Neutral'.
    """
    analysis = TextBlob(message)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return polarity, sentiment
