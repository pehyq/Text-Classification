# encoding=utf8
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
import simplejson as json
#import pickle
import numpy as np

def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>',re.S)
    return html_prog.sub('', str)

def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)

def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)

def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)

def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)

def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)

def replace_emoticon(emoticon_dict, str):
    for k, v in emoticon_dict.items():
        str = str.replace(k, v)
    return str

def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]','',current_tweet)


def pre_process(str, porter):
    # do not change the preprocessing order only if you know what you're doing 
    str = str.lower()
    str = rm_url(str)        
    str = rm_at_user(str)        
    str = rm_repeat_chars(str) 
    str = rm_hashtag_symbol(str)       
    str = rm_time(str)        
    str = rm_punctuation(str)
        
    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass
        
    return str
                            



if __name__ == "__main__":
    data_dir = './data'  ##Setting your own file path here.

    x_filename = 'samples.txt'
    y_filename = 'labels.txt'

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')


    ##load and process text data
    print('start loading and process text...')
    words_stat = {} # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding = "utf-8") as f:
        for i, line in enumerate(f):
            postprocess_tweet = []
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['text'].replace("\n"," ")
            words = pre_process(content, porter)
            for word in words:
                if word not in stops:
                    postprocess_tweet.append(word)
                    if word in words_stat.keys():
                        words_stat[word][0] += 1
                        if i != words_stat[word][2]:
                            words_stat[word][1] += 1
                            words_stat[word][2] = i
                    else:
                        words_stat[word] = [1,1,i]
            tweets.append(' '.join(postprocess_tweet))


            
    ##saving the statistics of tf and df for each words into file
    print("The number of unique words in data set is %i." %len(words_stat.keys()))
    lowTF_words = set()
    with open(os.path.join(data_dir, 'words_statistics.txt'), 'w', encoding = "utf-8") as f:
        f.write('TF\tDF\tWORD\n')
        for word, stat in sorted(words_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + word +  '\n')
            if stat[0]<2:
                lowTF_words.add(word)
    print("The number of low frequency words is %d." %len(lowTF_words))
    # print(stops)

    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'samples_processed.txt'), 'w', encoding = "utf-8")
    tweets_new = []
    for tweet in tweets:
        words = tweet.split(' ')
        new = [] 
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_tweet = ' '.join(new)
        tweets_new.append(new_tweet)
        fout.write('%s\n' %new_tweet)
    fout.close()

    print("Preprocessing text is completed")
   
    
    ## load and process User description
    print('start loading and process user description...')
    desc_stat = {} # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    desc = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding = "utf-8") as f:
        for i, line in enumerate(f):
            postprocess_desc = []
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['user']['description'].replace("\n"," ")
            words = pre_process(content, porter)
            for word in words:
                if word not in stops:
                    postprocess_desc.append(word)
                    if word in desc_stat.keys():
                        desc_stat[word][0] += 1
                        if i != desc_stat[word][2]:
                            desc_stat[word][1] += 1
                            desc_stat[word][2] = i
                    else:
                        desc_stat[word] = [1,1,i]
            desc.append(' '.join(postprocess_desc))

            
    ##saving the statistics of tf and df for each words into file
    print("The number of unique words in data set is %i." %len(desc_stat.keys()))
    lowTF_words = set()
    with open(os.path.join(data_dir, 'words_statistics_desc.txt'), 'w', encoding = "utf-8") as f:
        f.write('TF\tDF\tWORD\n')
        for word, stat in sorted(desc_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + word +  '\n')
            if stat[0]<2:
                lowTF_words.add(word)
    print("The number of low frequency words is %d." %len(lowTF_words))


    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'description_processed.txt'), 'w', encoding = "utf-8")
    tweets_new = []
    for tweet in desc:
        words = tweet.split(' ')
        new = [] 
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_tweet = ' '.join(new)
        tweets_new.append(new_tweet)
        fout.write('%s\n' %new_tweet)
    fout.close()
    
    print("Preprocessing description is completed")    
    
    
    ## Load and process hashtags
    print('start loading and process hashtag...')
    hashtags_stat = {} # record statistics of the df and tf for each hashtag; Form: {tag:[tf, df, tweet index]}
    hashtags = []
    cnt = 0      
    with open(os.path.join(data_dir, x_filename), encoding = "utf-8") as f:
        for i, line in enumerate(f):
            postprocess_hashtag_list = []
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            hashtag_list = tweet_obj['entities']['hashtags']
            no_of_hashtags = len(hashtag_list)
            hashtag_text_list = []
            if no_of_hashtags == 0:
                joined_postprocess_tags = ''
            else:
                for j in range(no_of_hashtags):
                    hashtag_text_list.append(hashtag_list[j]['text'])
                joined_tags = ' '.join(hashtag_text_list)
                tags = pre_process(joined_tags, porter)
                for tag in tags:
                    if tag not in stops:
                        postprocess_hashtag_list.append(tag)
                    if tag in hashtags_stat.keys():
                        hashtags_stat[tag][0] += 1
                        if i != hashtags_stat[tag][2]:
                            hashtags_stat[tag][1] += 1
                            hashtags_stat[tag][2] = i
                    else:
                        hashtags_stat[tag] = [1,1,i]
            joined_postprocess_tags = ' '.join(postprocess_hashtag_list)
            hashtags.append(joined_postprocess_tags)

    ## Save the statistics of tf and df for each hashtag into file
    print("The number of unique words in data set is %i." %len(hashtags_stat.keys()))
    lowTF_tags = set()
    with open(os.path.join(data_dir, 'hashtags_statistics.txt'), 'w', encoding = "utf-8") as f:
        f.write('TF\tDF\tHASHTAG\n')
        for tag, stat in sorted(hashtags_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + tag +  '\n')
            if stat[0]<2:
                lowTF_tags.add(tag)
    print("The number of low frequency words is %d." %len(lowTF_tags))
    
    ## Re-process samples, filter low frequency hashtags...
    fout = open(os.path.join(data_dir, 'hashtags_processed.txt'), 'w', encoding = "utf-8")
    new_hashtags_list = []
    for hashtag in hashtags:
        tags = hashtag.split(' ')
        new = [] 
        for tag in tags:
            if tag not in lowTF_tags:
                new.append(tag)
        new_hashtags = ' '.join(new)
        new_hashtags_list.append(new_hashtags)
        fout.write('%s\n' %new_hashtags)
    fout.close()

    print("Preprocessing hastag is completed")

    ##load and process geo data
    print('start loading and process geo data...')
    word_stat = {} # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    geo = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding = "utf-8") as f:
        for i, line in enumerate(f):
            postprocess_geo_list = []
            geo_words = []
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            user_loc = tweet_obj['user']['location'].replace("\n"," ")
            geo_words.append(user_loc)
            if tweet_obj['place'] != None:
                place_name = tweet_obj['place']['name'].replace("\n"," ")
                place_type = tweet_obj['place']['place_type'].replace("\n"," ")
                country = tweet_obj['place']['country'].replace("\n"," ")
                geo_words.append(place_name)
                geo_words.append(place_type)
                geo_words.append(country)
            for word in geo_words:
                if word not in stops:
                    postprocess_geo_list.append(word)
                    if word in words_stat.keys():
                        words_stat[word][0] += 1
                        if i != words_stat[word][2]:
                            words_stat[word][1] += 1
                            words_stat[word][2] = i
                    else:
                        words_stat[word] = [1,1,i]
            geo.append(' '.join(postprocess_geo_list))

            
    ##saving the statistics of tf and df for each words into file
    print("The number of unique words in data set is %i." %len(words_stat.keys()))
    lowTF_words = set()
    with open(os.path.join(data_dir, 'geo_statistics.txt'), 'w', encoding = "utf-8") as f:
        f.write('TF\tDF\tWORD\n')
        for word, stat in sorted(words_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + word +  '\n')
            if stat[0]<2:
                lowTF_words.add(word)
    print("The number of low frequency words is %d." %len(lowTF_words))
    # print(stops)

    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'geo_processed.txt'), 'w', encoding = "utf-8")
    geo_new = []
    for geox in geo:
        words = geox.split(' ')
        new = [] 
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_geo = ' '.join(new)
        geo_new.append(new_geo)
        fout.write('%s\n' %new_geo)
    fout.close()

    print("Preprocessing location data is completed")

    print("All Preprocessing completed")