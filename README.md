To set up an automated system that fetches data from MongoDB, processes and summarizes it using the OpenAI API, and posts summaries to a Discord channel, you will need to follow several steps. Here’s a detailed plan and the necessary scripts:

### Steps

1. **Set Up Environment**:
   - Install required libraries.
   - Set up MongoDB.
   - Set up OpenAI API.
   - Set up Discord bot.

2. **Fetch Data from MongoDB**:
   - Connect to MongoDB and fetch data every hour.

3. **Data Cleaning**:
   - Remove duplicates, tokenize, normalize text, remove stop words.

4. **Contextual and Semantic Filtering**:
   - Use NLP models to identify important posts.

5. **Generate Summaries Using OpenAI API**:
   - Summarize the filtered posts.

6. **Store Summaries in MongoDB**:
   - Save the generated summaries back to MongoDB.

7. **Post Summaries to Discord Channel**:
   - Use a Discord bot to post the summaries.

### Script

#### 1. Install Required Libraries

```bash
pip install pymongo discord.py openai nltk schedule
```

#### 2. Set Up MongoDB Connection

```python
from pymongo import MongoClient

def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")
    return client

def fetch_posts(client):
    db = client['social_media']
    collection = db['posts']
    posts = list(collection.find({"timestamp": {"$gte": <last_hour_timestamp>}}))
    return posts
```

#### 3. Data Cleaning

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
```

#### 4. Contextual and Semantic Filtering

```python
from transformers import pipeline

def filter_posts(posts):
    nlp_model = pipeline("sentiment-analysis")
    important_posts = []
    for post in posts:
        if any(keyword in post['text'].lower() for keyword in ['update', 'announcement', 'change', 'development']):
            sentiment = nlp_model(post['text'])[0]
            if sentiment['label'] == 'POSITIVE' or sentiment['label'] == 'NEGATIVE':
                important_posts.append(post)
    return important_posts
```

#### 5. Generate Summaries Using OpenAI API

```python
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_summary(post_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Summarize the following text: {post_text}",
        max_tokens=50
    )
    summary = response.choices[0].text.strip()
    return summary
```

#### 6. Store Summaries in MongoDB

```python
def store_summaries(client, summaries):
    db = client['social_media']
    collection = db['summaries']
    collection.insert_many(summaries)
```

#### 7. Post Summaries to Discord Channel

```python
import discord

client = discord.Client()

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

async def post_summary(channel_id, summary):
    channel = client.get_channel(channel_id)
    await channel.send(summary)

client.run('YOUR_DISCORD_BOT_TOKEN')
```

#### 8. Main Script to Orchestrate Everything

```python
import schedule
import time

def job():
    mongo_client = get_mongo_client()
    posts = fetch_posts(mongo_client)
    cleaned_posts = [clean_text(post['text']) for post in posts]
    important_posts = filter_posts(cleaned_posts)
    summaries = [{'post_id': post['_id'], 'summary': generate_summary(post['text'])} for post in important_posts]
    store_summaries(mongo_client, summaries)
    for summary in summaries:
        post_summary('YOUR_DISCORD_CHANNEL_ID', summary['summary'])

schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

### Detailed Steps Explanation

1. **Set Up Environment**:
   - Install all necessary Python libraries.
   - Set up MongoDB with collections for posts and summaries.
   - Register and set up a Discord bot.

2. **Fetch Data from MongoDB**:
   - Connect to MongoDB and fetch posts from the last hour.

3. **Data Cleaning**:
   - Clean the text data by removing duplicates, tokenizing, normalizing text, and removing stop words.

4. **Contextual and Semantic Filtering**:
   - Use NLP models to filter out posts that contain important keywords and have a significant sentiment.

5. **Generate Summaries Using OpenAI API**:
   - Summarize the filtered posts using the OpenAI API.

6. **Store Summaries in MongoDB**:
   - Save the generated summaries back into the MongoDB summaries collection.

7. **Post Summaries to Discord Channel**:
   - Use a Discord bot to post the summaries to a specific Discord channel.

By following these steps and using the provided scripts, you can set up an automated system to fetch, process, summarize, and post social media data.
