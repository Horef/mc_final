import time

from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import pandas as pd
from tqdm import tqdm

post_titles = []
post_urls = []
all_comments = []

class Comment:
    def __init__(self, comment_id, score, awards, length, replies=0):
        self.comment_id = comment_id
        self.score = score
        self.replies = replies
        self.awards = awards
        self.length = length
        self.length_to_avg_ratio = 0

    def add_reply(self):
        self.replies += 1

if __name__ == '__main__':
    driver = webdriver.Chrome()
    subreddits = ['macapps', 'learnprogramming', 'learntodraw', 'learnpython', 'learnmath', 'LaTeX', 'Python',
                  'datascience', 'dataengineering', 'malefashionadvice', 'MachineLearning', 'ObsidianMD',
                  'neuroscience', 'printSF', 'science', 'ios', 'MacOS', 'mac']

    # for each subreddit, we will get the posts and comments
    for subreddit in subreddits:
        print(f'Starting to scrape subreddit: {subreddit}')

        driver.get(f"https://www.reddit.com/r/{subreddit}/new/")
        time.sleep(3)

        # scrolling the page a few times to load more posts
        for i in tqdm(range(60), desc='Collecting Posts'):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            # finding all the articles on the page
            posts = BeautifulSoup(driver.page_source, 'html.parser').find_all('article', class_='w-full m-0')
            for post in posts:
                title = post['aria-label']
                post_box = post.find_all_next('shreddit-post')
                url = post_box[0]['content-href']
                # print(f'Post title: {title}')
                # print(f'Post URL: {url}\n')

                # if the title contains the question mark, it is a question post
                # we want to include only such posts
                # or, if it contains the word 'question' or 'help', we will include it
                if '?' in title or 'question' in title.lower() or 'help' in title.lower():
                    post_titles.append(title)
                    post_urls.append(url)

        print('Loading comments for each post')
        # in order to avoid repeating elements
        post_titles = list(set(post_titles))
        post_urls = list(set(post_urls))

        # for each post, we will open the URL and get the comments
        for i in tqdm(range(len(post_urls)), desc='Scraping Posts'):
            try:
                driver.get(post_urls[i])
            except Exception as e:
                print(f'Error: {e}')
                continue
            time.sleep(3)
            comments = BeautifulSoup(driver.page_source, 'html.parser').find_all('shreddit-comment')

            if len(comments) == 0:
                #print('No comments found for the post')
                continue

            # We are only interested in top-level comments, so for every comment, we will check its depth
            # if the depth is 0, we will include it in our analysis.
            # if the depth is 1, we find the corresponding top-level comment and increment its replies count
            # if the depth is 2 or more, we ignore it
            post_comments = []
            comment_length = 0
            for comment in comments:
                try:
                    comment_text_box = comment.find_all_next('div', attrs={'id' : '-post-rtjson-content'})
                    comment_text = comment_text_box[0].find_all_next('p')[0].text
                    # cleaning the comment text by removing new lines, tabs and carriage returns
                    comment_text = comment_text.replace('\n', '').replace('\t', '').replace('\r', '')
                    # removing leading and trailing whitespaces
                    comment_text = comment_text.strip()
                    comment_length += len(comment_text)

                    award_box = comment.find_all_next('award-button')
                    awards = award_box[0]['count']

                    # trying to find faceplate number of additional comments, if any
                    replies = 0
                    faceplate_box = comment.find_all_next('faceplate-partial', attrs={'slot': f'children-{comment['thingid']}-0'})
                    if len(faceplate_box) > 0:
                        faceplate_box_num = faceplate_box[0].find_all_next('faceplate-number')
                        if len(faceplate_box_num) > 0:
                            replies = int(faceplate_box_num[0]['number'])

                    if comment['depth'] == '0':
                        post_comments.append(Comment(comment_id=comment['thingid'],
                                                     score=comment['score'], awards=awards,
                                                     length=len(comment_text), replies=replies))
                    elif comment['depth'] == '1':
                        for prev_comment_id in range(len(post_comments)-1, 0, -1):
                            post_comment = post_comments[prev_comment_id]
                            if post_comment.comment_id == comment['parentid']:
                                post_comment.add_reply()
                                #print(f'Added reply to comment ID: {post_comment.comment_id}')
                except Exception as e:
                    print(f'Error: {e}')
                    continue

            # calculating the average length of the comments
            avg_comment_length = comment_length / len(post_comments)
            for comment in post_comments:
                comment.length_to_avg_ratio = comment.length / avg_comment_length
                #print(f'Comment ID: {comment.comment_id}, Score: {comment.score}, Replies: {comment.replies}, Awards: {comment.awards}, Length: {comment.length}, Length to Avg Ratio: {comment.length_to_avg_ratio}')

            all_comments.extend(post_comments)

        post_titles = []
        post_urls = []

    # creating a dataframe from the all_comments list
    df = pd.DataFrame([vars(comment) for comment in all_comments])
    # printing the amount of data we have collected
    print(f'Total number of comments collected: {len(df)}')

    # saving the dataframe to pickle file
    df.to_pickle('comments.pkl')

    driver.quit()
