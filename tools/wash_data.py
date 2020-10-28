# author: yx
# date: 2020/9/2 9:30
import json
import os
from tqdm import trange
from dateutil.parser import parse
from datetime import datetime
from matplotlib import pyplot as plt
from configs import cfg


def wash_data_4_nlpcc(cfg):
    """
    wash data for NLPCC 2018 Dataset
    :param cfg: config
    :return: summarization and article content of docs
    """
    file_name = cfg.DATASET.FLOW_DATA_CATALOG

    doc_info = []
    with open(file_name) as f_:
        line = f_.readline().strip('\r\n')
        while line:
            data = json.loads(line)
            summarization = data['summarization'].replace('\n', '')
            summarization = summarization.replace('\r', '')

            article = data['article'].replace('<Paragraph>', '')
            article = article.replace('\r', '')

            doc_info.append({
                'summarization': summarization.replace(' ', ''),
                'article': article.replace(' ', '')
                                 })
            line = f_.readline().strip('\r\n')

    return doc_info

def wash_data_4_sohu_news(cfg):
    """
    wash data for sohu news dataset
    :param cfg: config
    :return: time, title and article content of news
    """
    file_name = cfg.DATASET.SOGOU_NEWS_CORPUS
    doc_info = []

    # after data analysis, we find the news distribution concentrated in 2012.6 - 2012.8, so only extract these data
    min_datetime = datetime(2012, 5, 31)
    max_datetime = datetime(2012, 8, 1)

    with open(file_name, 'r') as f:
        lines = f.readlines()
        '''
        data format of sogou news dataset is like:
        
        <doc>
        <url>http://gongyi.sohu.com/20120619/n346051128.shtml</url>
        <docno>48d8394cb8d2f0ea-34913306c0bb3300</docno>
        <contenttitle>title content</contenttitle>
        <content>article content</content>
        </doc>
        
        so time at line 1(in url), title at line 3 and content at line 4 for each doc  
        '''
        print('data loading:')
        for i in trange(0, len(lines), 6):
            url = lines[i + 1]
            time = url.split('/')[3]
            if time.isdigit():
                try:
                    time = parse(time)

                    title = lines[i + 3].replace('<contenttitle>', '')
                    title = title.replace('</contenttitle>', '')
                    title = title.replace('\n', '')
                    title = title.replace('\r', '')


                    content = lines[i + 4].replace('<content>', '')
                    content = content.replace('</content>', '')
                    content = content.replace('\n', '')
                    content = content.replace('\r', '')
                    content = content.replace(' ', '')

                    # # only extract date in 2012.6 - 2012.8, news number is 960825
                    # if min_datetime < time < max_datetime and content != '':
                    #     doc_info.append({
                    #         'time': time,
                    #         'title': title,
                    #         'content': content
                    #     })
                    # pass

                    # only extract content contain '火灾' or '大火'
                    if '火灾' in content or '大火' in content:
                        doc_info.append({
                            'time': time,
                            'title': title,
                            'content': content
                        })
                    pass
                except:
                    continue
                pass
            pass
        pass
    return doc_info

def split_sohu_news_by_day(cfg):
    """
    split daily news
    :param cfg: config
    :return:
    """
    doc_info = wash_data_4_sohu_news(cfg=cfg)

    # create daily news catalog if the catalog is not exist
    # flow_data_catalog = cfg.DATASET.FLOW_DATA_CATALOG
    flow_data_catalog = cfg.DATASET.FLOW_DATA_CATALOG_FIRE

    if flow_data_catalog and not os.path.exists(flow_data_catalog):
        os.mkdir(flow_data_catalog)
        pass

    dates = {}
    print('data washing:')
    for i in trange(len(doc_info)):
        daily_news_file = os.path.join(flow_data_catalog, doc_info[i]['time'].strftime("%Y%m%d")) + '.txt'
        try:
            dates[doc_info[i]['time']] += 1
            with open(daily_news_file, 'a+', encoding='utf-8') as f:
                f.write(doc_info[i]['content'])
                f.write('\n')
            pass
        except:
            dates[doc_info[i]['time']] = 1

            if not os.path.exists(daily_news_file):
                with open(daily_news_file, 'a+', encoding='utf-8') as f:
                    f.write(doc_info[i]['content'])
                    f.write('\n')
            pass
        pass

    # show the num of news distribution
    time = [date for date in dates.keys()]
    num = [count for count in dates.values()]
    plt.plot_date(time, num)
    plt.savefig('daily_news_distribution_fire.jpg')
    plt.show()
    pass

if __name__ == '__main__':
    split_sohu_news_by_day(cfg)
    pass


