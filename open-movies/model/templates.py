import csv
import pandas as pd

class Templates(object):
    def __init__(self, titles, genres, templates):
        tables = {}
        tables['titles'] = pd.DataFrame(titles)
        tables['genres'] = pd.DataFrame(genres)
        tables['templates'] = pd.DataFrame(templates)
        self.tables = tables

    @classmethod
    def from_pickle(cls, path):
        templates = read_pickle(path)
        return cls(templates)

    @classmethod
    def from_csv(cls, path):
        titles = []
        genres = []
        templates = []
        # 1st sentence, last sentence, middle sentence
        position_map = lambda i: 0 if i == 0 else -1 if i == 1 else 1
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                id_ = i
                titles.append({'id': id_, 'title': row['title']})
                movie_genres = eval(row['genres']) if row['genres'] else []
                genres.extend([{'id': id_, 'genre': x['name']} for x in movie_genres])
                utterances = eval(row['templates']) if row['templates'] else []
                templates.extend([{'id': id_, 'utterance': x, 'pos': position_map(j)} for j, x in enumerate(utterances)])
        return cls(titles, genres, templates)

if __name__ == '__main__':
    templates = Templates.from_csv('/juicier/scr105/scr/derekchen14/movie_data/joined/combined.csv')
    print set(templates.tables['genres']['genre'].values)
