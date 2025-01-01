from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import tempfile
from dotenv import load_dotenv
import logging
from datetime import datetime
from collections import Counter
import traceback
import sys
from math import ceil
import html

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create temp directory
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
tempfile.tempdir = temp_dir

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
load_dotenv()

app = Flask(__name__)

# Initialize YouTube API client
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    logger.error("YouTube API key not found in environment variables")
    raise ValueError("YouTube API key is required")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Add these constants near the top
VIDEOS_PER_PAGE = 5
MAX_PAGES = 10  # Maximum number of pages to prevent excessive API usage

def get_video_stats(video_id):
    try:
        stats = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()
        return stats['items'][0]['statistics']
    except Exception as e:
        logger.error(f"Error fetching video stats: {str(e)}")
        return {}

def get_video_comments(video_id):
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < 1000:  # Keep fetching until we have 1000 comments
            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,  # Maximum allowed per request
                    textFormat='plainText',
                    order='relevance',
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                if not response.get('items'):
                    break
                    
                for item in response['items']:
                    try:
                        comment_data = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'text': comment_data['textDisplay'],
                            'likes': comment_data.get('likeCount', 0),
                            'date': comment_data['publishedAt']
                        })
                    except KeyError as e:
                        logger.warning(f"Skipping malformed comment: {str(e)}")
                        continue
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching comment page: {str(e)}")
                break
                
        logger.info(f"Retrieved {len(comments)} comments for video {video_id}")
                
    except HttpError as e:
        if e.resp.status == 403:
            logger.warning(f"Comments are disabled for video {video_id}")
        else:
            logger.error(f"Error fetching comments for {video_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching comments for {video_id}: {str(e)}")
    
    return comments

def analyze_sentiment(comments):
    if not comments:
        return {
            'average_sentiment': 0,
            'sentiment_score': 0,  # New 0-100 score
            'total_comments': 0,
            'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
            'example_comments': {},
            'engagement_metrics': {},
            'status': 'no_comments'
        }
        
    try:
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        sentiment_examples = {'positive': [], 'negative': [], 'neutral': []}
        total_likes = 0
        
        # Analyze each comment
        for comment in comments:
            if not isinstance(comment['text'], str):
                continue
                
            sentiment_score = sia.polarity_scores(comment['text'])
            compound_score = sentiment_score['compound']
            sentiments.append(compound_score)
            total_likes += comment['likes']
            
            # Store example comments with more extreme thresholds
            if compound_score >= 0.6:  # More strict positive threshold
                sentiment_examples['positive'].append({
                    'text': comment['text'],
                    'score': compound_score,
                    'likes': comment['likes'],
                    'date': comment['date']
                })
            elif compound_score <= -0.6:  # More strict negative threshold
                sentiment_examples['negative'].append({
                    'text': comment['text'],
                    'score': compound_score,
                    'likes': comment['likes'],
                    'date': comment['date']
                })
            elif -0.2 <= compound_score <= 0.2:  # Stricter neutral threshold
                sentiment_examples['neutral'].append({
                    'text': comment['text'],
                    'score': compound_score,
                    'likes': comment['likes'],
                    'date': comment['date']
                })
        
        # Sort examples by likes and get top 3 for each category
        for category in sentiment_examples:
            sentiment_examples[category] = sorted(
                sentiment_examples[category],
                key=lambda x: x['likes'],
                reverse=True
            )[:3]
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Convert -1 to 1 scale to 0-100 scale
        sentiment_score = int((avg_sentiment + 1) * 50)  # Convert to 0-100 scale
        
        # Calculate weighted sentiment based on likes
        positive_count = len([s for s in sentiments if s > 0])
        negative_count = len([s for s in sentiments if s < 0])
        neutral_count = len([s for s in sentiments if s == 0])
        
        return {
            'average_sentiment': avg_sentiment,
            'sentiment_score': sentiment_score,  # New 0-100 score
            'sentiment_description': get_sentiment_description(sentiment_score),  # New function
            'total_comments': len(comments),
            'sentiment_breakdown': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'example_comments': sentiment_examples,
            'engagement_metrics': {
                'total_likes': total_likes,
                'avg_likes_per_comment': total_likes / len(comments) if comments else 0
            },
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {
            'status': 'analysis_error'
        }

def get_sentiment_description(score):
    """Return a human-readable description of the sentiment score"""
    if score >= 90:
        return "Overwhelmingly Positive"
    elif score >= 75:
        return "Very Positive"
    elif score >= 60:
        return "Positive"
    elif score >= 40:
        return "Mixed"
    elif score >= 25:
        return "Negative"
    elif score >= 10:
        return "Very Negative"
    else:
        return "Overwhelmingly Negative"

# Add new function to get channel details
def get_channel_stats(channel_id):
    try:
        response = youtube.channels().list(
            part='statistics,snippet',
            id=channel_id
        ).execute()
        
        if response['items']:
            return response['items'][0]
        return None
    except Exception as e:
        logger.error(f"Error fetching channel stats: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    page = int(request.form.get('page', 1))
    search_type = request.form.get('type', 'video')
    
    if not query:
        return jsonify({'error': 'Please enter a search term'}), 400
    
    try:
        logger.info(f"Searching for {search_type}: {query}, page: {page}")
        
        next_page_token = request.form.get('next_page_token') if page > 1 else None

        # Search for videos and/or channels
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=VIDEOS_PER_PAGE,
            type=search_type,
            pageToken=next_page_token
        ).execute()
        
        results = []
        
        for item in search_response['items']:
            try:
                if search_type == 'video' and item['id'].get('videoId'):
                    video_id = item['id']['videoId']
                    video_data = {
                        'type': 'video',
                        'id': video_id,
                        'title': html.unescape(item['snippet']['title']),
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'description': html.unescape(item['snippet']['description']),
                        'channel_title': html.unescape(item['snippet']['channelTitle']),
                        'published_at': item['snippet']['publishedAt']
                    }
                    
                    # Get video statistics
                    try:
                        stats = get_video_stats(video_id)
                        video_data['statistics'] = stats or {
                            'viewCount': '0',
                            'likeCount': '0',
                            'commentCount': '0'
                        }
                    except Exception as e:
                        logger.error(f"Error getting video stats: {str(e)}")
                        video_data['statistics'] = {
                            'viewCount': '0',
                            'likeCount': '0',
                            'commentCount': '0'
                        }
                    
                    # Get comments and analyze sentiment
                    try:
                        comments = get_video_comments(video_id)
                        sentiment_analysis = analyze_sentiment(comments)
                        video_data['sentiment'] = sentiment_analysis
                    except Exception as e:
                        logger.error(f"Error analyzing comments: {str(e)}")
                        video_data['sentiment'] = {
                            'status': 'error',
                            'message': 'Could not analyze comments',
                            'average_sentiment': 0,
                            'sentiment_score': 50,
                            'total_comments': 0,
                            'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
                            'example_comments': {'positive': [], 'negative': [], 'neutral': []},
                            'engagement_metrics': {'total_likes': 0, 'avg_likes_per_comment': 0}
                        }
                    
                    results.append(video_data)
                    
                elif search_type == 'channel' and item['id'].get('channelId'):
                    channel_id = item['id']['channelId']
                    channel_stats = get_channel_stats(channel_id)
                    
                    if channel_stats:
                        channel_data = {
                            'type': 'channel',
                            'id': channel_id,
                            'title': html.unescape(item['snippet']['title']),
                            'description': html.unescape(item['snippet']['description']),
                            'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                            'statistics': channel_stats['statistics'],
                            'customUrl': html.unescape(channel_stats['snippet'].get('customUrl', '')),
                            'publishedAt': item['snippet']['publishedAt']
                        }
                        results.append(channel_data)
                        
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}")
                continue

        return jsonify({
            'results': results,
            'has_more': 'nextPageToken' in search_response,
            'next_page_token': search_response.get('nextPageToken')
        })
        
    except HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error("API quota exceeded")
            return jsonify({
                'error': 'API quota exceeded',
                'details': 'Daily YouTube API limit reached. Please try again tomorrow or use a different API key.'
            }), 429  # 429 Too Many Requests
        else:
            logger.error(f"YouTube API error: {str(e)}")
            return jsonify({
                'error': 'YouTube API error',
                'details': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

def test_api_key():
    try:
        test_response = youtube.search().list(
            q='test',
            part='id',
            maxResults=1
        ).execute()
        logger.info("✓ YouTube API key is valid")
        return True
    except Exception as e:
        logger.error(f"✗ YouTube API key test failed: {str(e)}")
        return False

@app.route('/channel_videos', methods=['POST'])
def channel_videos():
    channel_id = request.form.get('channel_id')
    page_token = request.form.get('page_token')
    
    if not channel_id:
        return jsonify({'error': 'Channel ID is required'}), 400
        
    try:
        # Increased maxResults to 50 videos per request (maximum allowed)
        search_response = youtube.search().list(
            channelId=channel_id,
            part='id,snippet',
            maxResults=50,  # Increased from 25 to 50 (YouTube API maximum)
            type='video',
            order='date',
            pageToken=page_token
        ).execute()
        
        videos = []
        
        for item in search_response['items']:
            try:
                if item['id'].get('videoId'):
                    video_id = item['id']['videoId']
                    video_data = {
                        'type': 'video',
                        'id': video_id,
                        'title': html.unescape(item['snippet']['title']),
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'description': html.unescape(item['snippet']['description']),
                        'channel_title': html.unescape(item['snippet']['channelTitle']),
                        'published_at': item['snippet']['publishedAt']
                    }
                    
                    # Get video statistics
                    try:
                        stats = get_video_stats(video_id)
                        video_data['statistics'] = stats or {
                            'viewCount': '0',
                            'likeCount': '0',
                            'commentCount': '0'
                        }
                    except Exception as e:
                        logger.error(f"Error getting video stats: {str(e)}")
                        video_data['statistics'] = {
                            'viewCount': '0',
                            'likeCount': '0',
                            'commentCount': '0'
                        }
                    
                    # Get comments and analyze sentiment
                    try:
                        comments = get_video_comments(video_id)
                        sentiment_analysis = analyze_sentiment(comments)
                        video_data['sentiment'] = sentiment_analysis
                    except Exception as e:
                        logger.error(f"Error analyzing comments: {str(e)}")
                        video_data['sentiment'] = {
                            'status': 'error',
                            'message': 'Could not analyze comments',
                            'average_sentiment': 0,
                            'sentiment_score': 50,
                            'total_comments': 0,
                            'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
                            'example_comments': {'positive': [], 'negative': [], 'neutral': []},
                            'engagement_metrics': {'total_likes': 0, 'avg_likes_per_comment': 0}
                        }
                    
                    videos.append(video_data)
                    
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                continue
                
        return jsonify({
            'videos': videos,
            'has_more': 'nextPageToken' in search_response,
            'next_page_token': search_response.get('nextPageToken')
        })
        
    except Exception as e:
        logger.error(f"Error fetching channel videos: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    if not test_api_key():
        logger.error("Invalid YouTube API key. Please check your .env file and Google Cloud Console settings.")
        sys.exit(1)
    app.run(debug=True) 