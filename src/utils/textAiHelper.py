from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class Turtle:
    @staticmethod
    def clean_data(text):
        """
        :param text: dirty text
        :return: cleaned, stemmed text
        """
        stemmer = PorterStemmer()

        try:
            clean_text = text.str.lower()
            clean_text = clean_text.str.replace('\d+', '')
            clean_text = clean_text.str.strip()
            clean_text = clean_text.str.replace('[^\w\s]', ' ')
            clean_text = clean_text.str.replace('br', '')
            clean_text = clean_text.str.replace(' +', ' ')
            clean_text = clean_text.str.replace('\d+', '')

            stop = stopwords.words('english')
            stop.extend(["movie", "movies", "film", "one"])
            clean_text = clean_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
            # Stemming
            clean_text = clean_text.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))

            return clean_text
        except Exception as e:
            print("In Exception of clean_data: ", e)
            return None

    @staticmethod
    def tokenization(df_reviews):
        """
        :param df_reviews: reviews
        :return: tokenized text
        """
        count_vec = CountVectorizer(analyzer='word', tokenizer=lambda doc: doc, lowercase=False, max_df=0.70,
                                    min_df=100)
        print(" Tokenize the Reviews")

        df_reviews["Clean_Review"] = df_reviews["Clean_Review"].astype(str).str.strip().str.split('[\W_]+')

        words_vec = count_vec.fit(df_reviews["Clean_Review"])
        bag_of_words = words_vec.transform(df_reviews["Clean_Review"])
        tokens = count_vec.get_feature_names()
        df_words = pd.DataFrame(data=bag_of_words.toarray(), columns=tokens)
        return df_words

