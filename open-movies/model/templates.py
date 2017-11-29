import csv
import random
import pandas as pd
import numpy as np
import json

from cocoa.core.util import generate_uuid, write_pickle, read_pickle

class Templates(object):
    def __init__(self, titles, genres, templates):
        tables = {}
        tables['titles'] = pd.DataFrame(titles)
        tables['genres'] = pd.DataFrame(genres)
        tables['templates'] = pd.DataFrame(templates)
        self.tables = tables

    def save_pickle(self, path):
        print 'Dump templates to {}'.format(path)
        write_pickle(self.tables, path)

    @classmethod
    def from_pickle(cls, path):
        data = read_pickle(path)
        return cls(**data)

    def known_movies(self):
        """Return a set of movies that we have templates for.
        """
        df = self.tables['titles']
        movie_ids = set(self.tables['templates']['movie_id'].values)
        titles = df.loc[df.movie_id.isin(movie_ids)]['title'].values
        return set(titles)

    def known_popular_movies(self):
        #known = self.known_movies()
        df = self.tables['templates']
        sources = set(('bestof2016', 'bestof2017', 'classics', 'manual'))
        movie_ids = set(df.loc[df.source.isin(sources)]['movie_id'].values)
        df = self.tables['titles']
        popular = df.loc[df.movie_id.isin(movie_ids)]['title'].values
        return popular

    @classmethod
    def read_kaggle_reviews(cls, path):
        titles = []
        genres = []
        templates = []
        # first sentence, last sentence, middle sentence
        # TODO: more complex tag (e.g. sentiment, plot, see Yoav's style paper)
        tag_review = lambda pos: 'first' if pos == 0 else 'last' if pos == 1 else 'middle'
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                id_ = i
                titles.append({'movie_id': id_, 'title': row['title']})
                movie_genres = eval(row['genres']) if row['genres'] else []
                genres.extend([{'movie_id': id_, 'genre': x['name']} for x in movie_genres])
                utterances = eval(row['templates']) if row['templates'] else []
                templates.extend([{
                    'id': generate_uuid('T'),
                    'movie_id': id_,
                    'template': x,
                    'tag': 'inform-{}'.format(tag_review(j)),
                    'source': 'kaggle',
                    'context_tag': 'ask'
                    }
                    for j, x in enumerate(utterances) if
                    len(x.split()) < 20])
        return titles, genres, templates

    @classmethod
    def read_rotten_reviews(cls, path):
        titles = []
        genres = []
        templates = []
        movies = json.load(open(path, "r"))
        for i, row in enumerate(movies):
            id_ = i
            titles.append({'movie_id': id_, 'title': row['title']})
            for tag, utterances in row['templates'].items():
                templates.extend([{
                    'id': generate_uuid('T'),
                    'movie_id': id_,
                    'template': x,
                    'tag': 'inform-{}'.format(tag),
                    'source': row['source'],
                    'context_tag': 'ask'
                    }
                    for x in utterances if len(x.split()) < 28])

        return titles, genres, templates

    @classmethod
    def read_templates(cls, path):
        """Read handcoded templates.

        csv header: tag, context_tag, template

        """
        templates = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                data = {k: None if v == '' else v for k, v in row.iteritems()}
                data['id'] = generate_uuid('T')
                data['source'] = 'handcoded'
                templates.append(data)
        return templates

    @classmethod
    def from_file(cls, review_path, handcoded_path, source):
        if source == "rotten":
            titles, genres, templates = cls.read_rotten_reviews(review_path)
        elif source == "kaggle":
            titles, genres, templates = cls.read_kaggle_reviews(review_path)
        handcoded_templates = cls.read_templates(handcoded_path)
        templates.extend(handcoded_templates)
        return cls(titles, genres, templates)


    def search(self, context_tag=None, movie_title=None, used_templates=None, tag=None):
        loc = self.get_filter(context_tag=context_tag, movie_title=movie_title, used_templates=used_templates, tag=tag)
        templates = self.tables['templates'].loc[loc]
        if len(templates) == 0:
            return None
        template_id = random.randint(0, len(templates)-1)
        return templates.iloc[template_id]

    def get_filter(self, context_tag=None, movie_title=None, used_templates=None, tag=None):
        locs = [True]
        def add_filter(cond):
            locs.append(locs[-1] & cond)
        templates = self.tables['templates']
        if used_templates:
            add_filter(~templates.id.isin(used_templates))
            if np.sum(locs[-1]) == 0:
                del locs[-1]
        if tag:
            add_filter(templates.tag == tag)
        if context_tag:
            add_filter(templates.context_tag == context_tag)
        if movie_title:
            movie_id = self.get_movie_id(movie_title)
            if movie_id:
                add_filter(templates.movie_id == movie_id)
        for loc in locs[:0:-1]:
            if np.sum(loc) > 0:
                return loc
        return locs[1]

    def get_movie_id(self, title):
        df = self.tables['titles']
        results = df.loc[df.title == title]['movie_id'].values
        return results[0] if len(results) > 0 else None


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # /juicier/scr105/scr/derekchen14/movie_data/all_merged.json
    parser.add_argument('--movie-data', help='Path to movie metadata and reviews')
    # /juicy/scr61/scr/nlp/hehe/cocoa/open-movie/data/handcoded_templates.csv
    parser.add_argument('--templates', help='Path to handcoded templates')
    parser.add_argument('--output', help='Path to save templates')
    parser.add_argument('--source', choices=["kaggle", "rotten"], default="rotten",
        help='kaggle (which is The Movie DB) or rotten (which is Rotten Tomatoes)')
    args = parser.parse_args()

    templates = Templates.from_file(args.movie_data, args.templates, args.source)
    templates.save_pickle(args.output)

    # template = templates.search(movie_title='11 Harrowhouse')
    template = templates.search(movie_title='finding dory')
    print template['template']
    import sys; sys.exit()

