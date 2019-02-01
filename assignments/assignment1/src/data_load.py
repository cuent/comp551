import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
from data_process import generate_data


def read_data():
    '''Open the original json file with the data'''
    # popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
    # children : the number of replies to this comment (type: int)
    # text : the text of this comment (type: string)
    # controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
    # is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment

    with open("../data/reddit.json") as fp:
        data = json.load(fp)
    return data


if __name__ == '__main__':
    stopwords = open('../data/stopwords.txt').read().splitlines()

    data = read_data()
    # Generate data according w/section 3.1/3.2
    generate_data(data, ['text', 'is_root', 'controversiality', 'children'], "default_top160", top_words=160)
    generate_data(data, ['text', 'is_root', 'controversiality', 'children'], "default_top60", top_words=60)
    generate_data(data, ['is_root', 'controversiality', 'children'], "default_notext")

    # Generate new features
    generate_data(data, ['is_root', 'controversiality', 'children', 'text'], "default_tfidf", dictType='tfidf')
    generate_data(data, ['is_root', 'controversiality', 'children', 'text'], "default_feeling", dictType='feelings')
    generate_data(data, ['is_root', 'children', 'text'], "default_noroot")
    generate_data(data, ['children', 'text'], "only_children_text")
    generate_data(data, ['children'], "only_children")
    generate_data(data, ['children', 'square_children'], "only_children_square")
    generate_data(data, ['children', 'square_children', 'text'], "square")
    generate_data(data, ['children', 'square_children', 'cube_children'], "only_cube")
    generate_data(data, ['children', 'square_children', 'cube_children', 'fourth_children'], "only_fourth")
    generate_data(data, ['children', 'text'], "stopwords_children_120", top_words=120, stop_words=stopwords)
    generate_data(data, ['children', 'text'], "stopwords_children_50", top_words=50, stop_words=stopwords)
    generate_data(data, ['children', 'text', 'len_text'], "stopwords_len_top50", top_words=50, stop_words=stopwords)
    generate_data(data, ['children', 'text', 'len_text'], "len_top50", top_words=50, )
    generate_data(data,
                  ['children', 'text', 'len_text', 'len_sentence', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
                   'sentiment_compound'], "children_sentiment_top10", top_words=10)
    generate_data(data,
                  ['children', 'len_text', 'len_sentence', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
                   'sentiment_compound'], "children_sentiment")
    generate_data(data, [
        'children',
        'square_children',
        'len_text',
        'sentiment_neg',
        'sentiment_neu',
        'sentiment_pos',
        'text'
    ], "most_important_features",
                  # stop_words=stopwords,
                  top_words=57)

    # two best models
    generate_data(data, ['text', 'is_root', 'controversiality', 'children', 'len_text'], "lenght_text",
                  top_words=62)
    generate_data(data, ['text', 'is_root', 'controversiality', 'children', 'square_children'], "children_all",
                  top_words=57)
    generate_data(data, ['text', 'is_root', 'controversiality', 'children', 'len_text', 'square_children'],
                  "best_combination", top_words=57)
    generate_data(data, ['text', 'is_root', 'controversiality', 'children', 'len_text', 'square_children'],
                  "best_combination1", top_words=62)
    generate_data(data, ['text', 'is_root', 'controversiality', 'children', 'len_text', 'square_children'],
                  "best_combination2", top_words=60)

    # Some experiments to find best features. Only the best selected.
    # # Naive implementation for searching features
    # f = [
    #     'text',
    #     'is_root', 'controversiality', 'children',
    #     # 'square_children',
    #     'len_text',
    #     # 'len_sentence', 'sentiment_neg',
    #     # 'sentiment_neu',
    #     # 'sentiment_pos',
    #     # 'sentiment_compound'
    # ]
    #
    # # import     itertools
    # # i = 0
    # # for L in range(1, len(f) + 1):
    # #     for subset in itertools.combinations(f, L):
    # #         print(i, list(subset))
    # #         generate_data(data, list(subset), "test"+str(i))
    # #         i+=1
    #
    # # Vary words
    #
    # f = [
    #     # ['text', 'is_root', 'children', 'square_children'],
    #     # ['text', 'controversiality', 'children', 'square_children'],
    #     ['text', 'is_root', 'controversiality', 'children', 'square_children'],  # 1
    #     # ['text', 'is_root', 'controversiality', 'children', 'sentiment_neg'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'sentiment_neu'],  # 2
    #     # ['text', 'is_root', 'controversiality', 'children', 'sentiment_pos'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'len_text'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'len_sentence'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'cube_children', 'fourth_children'],  # *
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'fourth_children'],  # *
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'cube_children', 'fourth_children'],  # *
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'len_text'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_neg'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_neu'],  # 3
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_pos'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'len_text', 'sentiment_neg'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'len_text', 'sentiment_neu'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'len_text', 'sentiment_pos'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_neg', 'sentiment_neu'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_neg', 'sentiment_pos'],
    #     # ['text', 'is_root', 'controversiality', 'children', 'square_children', 'sentiment_neu', 'sentiment_pos']
    # ]
    #
    # # i = 0
    # # for l in f:
    # #     print(i, l)
    # #     generate_data(data, l, "test" + str(i), top_words=57)
    # #     i += 1
