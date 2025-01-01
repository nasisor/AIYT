import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
import html

# Page config
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Download NLTK data
@st.cache_resource
def download_nltk():
    nltk.download('vader_lexicon', quiet=True)

download_nltk()

# Initialize YouTube API client
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Helper functions (reuse from your Flask app)
def get_video_stats(video_id):
    try:
        stats = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()
        return stats['items'][0]['statistics']
    except Exception as e:
        st.error(f"Error fetching video stats: {str(e)}")
        return {}

def get_video_comments(video_id):
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < 1000:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                textFormat='plainText',
                order='relevance',
                pageToken=next_page_token
            )
            
            response = request.execute()
            
            if not response.get('items'):
                break
                
            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment_data['textDisplay'],
                    'likes': comment_data.get('likeCount', 0),
                    'date': comment_data['publishedAt']
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
    
    return comments

def analyze_sentiment(comments):
    if not comments:
        return {
            'sentiment_score': 50,
            'total_comments': 0,
            'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
            'example_comments': {'positive': [], 'negative': [], 'neutral': []},
            'engagement_metrics': {'total_likes': 0, 'avg_likes_per_comment': 0}
        }
    
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    sentiment_examples = {'positive': [], 'negative': [], 'neutral': []}
    total_likes = 0
    
    for comment in comments:
        sentiment_score = sia.polarity_scores(comment['text'])
        compound_score = sentiment_score['compound']
        sentiments.append(compound_score)
        total_likes += comment['likes']
        
        if compound_score >= 0.6:
            sentiment_examples['positive'].append(comment)
        elif compound_score <= -0.6:
            sentiment_examples['negative'].append(comment)
        elif -0.2 <= compound_score <= 0.2:
            sentiment_examples['neutral'].append(comment)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    sentiment_score = int((avg_sentiment + 1) * 50)
    
    return {
        'sentiment_score': sentiment_score,
        'total_comments': len(comments),
        'sentiment_breakdown': {
            'positive': len([s for s in sentiments if s > 0]),
            'negative': len([s for s in sentiments if s < 0]),
            'neutral': len([s for s in sentiments if s == 0])
        },
        'example_comments': sentiment_examples,
        'engagement_metrics': {
            'total_likes': total_likes,
            'avg_likes_per_comment': total_likes / len(comments) if comments else 0
        }
    }

# Streamlit UI
st.title("YouTube Sentiment Analyzer")
st.markdown("Analyze the sentiment of YouTube video comments using AI")

# Search input
search_type = st.selectbox("Search Type", ["Videos", "Channels"], key="search_type")
query = st.text_input("Search YouTube", key="search_input")

if st.button("Analyze"):
    if not query:
        st.error("Please enter a search term")
    else:
        with st.spinner("Searching..."):
            try:
                # Search for videos or channels
                search_response = youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=10,
                    type=search_type.lower()[:-1]  # Remove 's' from end
                ).execute()
                
                if not search_response.get('items'):
                    st.warning("No results found")
                else:
                    for item in search_response['items']:
                        col1, col2 = st.columns([1, 2])
                        
                        if search_type == "Videos":
                            video_id = item['id']['videoId']
                            with col1:
                                st.image(item['snippet']['thumbnails']['medium']['url'])
                            
                            with col2:
                                st.subheader(html.unescape(item['snippet']['title']))
                                st.write(f"Channel: {html.unescape(item['snippet']['channelTitle'])}")
                                
                                # Get video stats and analyze sentiment
                                stats = get_video_stats(video_id)
                                comments = get_video_comments(video_id)
                                sentiment = analyze_sentiment(comments)
                                
                                # Display metrics
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Sentiment Score", f"{sentiment['sentiment_score']}/100")
                                with metrics_col2:
                                    st.metric("Total Comments", sentiment['total_comments'])
                                with metrics_col3:
                                    st.metric("Avg. Likes per Comment", 
                                            f"{sentiment['engagement_metrics']['avg_likes_per_comment']:.1f}")
                                
                                # Show example comments in tabs
                                if sentiment['example_comments']['positive'] or \
                                   sentiment['example_comments']['negative'] or \
                                   sentiment['example_comments']['neutral']:
                                    tab1, tab2, tab3 = st.tabs(["Positive", "Negative", "Neutral"])
                                    
                                    with tab1:
                                        for comment in sentiment['example_comments']['positive'][:3]:
                                            st.text(html.unescape(comment['text']))
                                    
                                    with tab2:
                                        for comment in sentiment['example_comments']['negative'][:3]:
                                            st.text(html.unescape(comment['text']))
                                            
                                    with tab3:
                                        for comment in sentiment['example_comments']['neutral'][:3]:
                                            st.text(html.unescape(comment['text']))
                        
                        else:  # Channels
                            channel_id = item['id']['channelId']
                            with col1:
                                st.image(item['snippet']['thumbnails']['medium']['url'])
                            
                            with col2:
                                st.subheader(html.unescape(item['snippet']['title']))
                                st.write(html.unescape(item['snippet']['description']))
                                
                                # Get channel stats
                                channel_stats = youtube.channels().list(
                                    part='statistics',
                                    id=channel_id
                                ).execute()
                                
                                if channel_stats.get('items'):
                                    stats = channel_stats['items'][0]['statistics']
                                    st.metric("Subscribers", f"{int(stats.get('subscriberCount', 0)):,}")
                                    st.metric("Total Videos", f"{int(stats.get('videoCount', 0)):,}")
                                    st.metric("Total Views", f"{int(stats.get('viewCount', 0)):,}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • [View on GitHub](https://github.com/yourusername/youtube-sentiment)") 