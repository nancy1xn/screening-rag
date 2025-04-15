from screening_rag.crawlers.cnn_crime_event_pipeline import get_cnn_news, SortingBy

def test_get_cnn_news():

    #測試時間順序
    news_list =[]
    for news in get_cnn_news("putin", 3, SortingBy.NEWEST):
        news_list.append(news)
    assert news_list[0].date_publish > news_list[1].date_publish
    assert news_list[1].date_publish > news_list[2].date_publish
    # assert news_list[14].date_publish > news_list[15].date_publish
    # assert news_list[18].date_publish> news_list[19].date_publish
    # assert news_list[0].date_publish > news_list[19].date_publish

    #測試關鍵字是否存在maintext
    for news in get_cnn_news("putin", 3, SortingBy.RELEVANCY):
        assert "putin" in news.maintext.lower()
    
    #測試長度
    news_list =[]
    for news in get_cnn_news("putin", 3, SortingBy.RELEVANCY):
        news_list.append(news)
    assert len(news_list) ==3 