from nltk.stem import SnowballStemmer

def run():
    stemmer = SnowballStemmer('german')
    word = 'Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz'
    stemmed_word = stemmer.stem(word)
    print(f"Original word: {word}, Stemmed word: {stemmed_word}")


if __name__ == '__main__':
    run()
