import argparse
import collections
import editdistance
import json
import numpy as np
import re
import sklearn

from fuzzywuzzy import fuzz
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from stop_words import get_stop_words


class EntityRanker(object):
    """
    Learned ranker for ranking candidates of a span of text for the lexicon
    """
    def __init__(self, entity_annotations, scenarios, train_data, transcripts_infile):
        """
        :param entity_annotations:  Path to JSON of entity annotations
        :param scenarios: Path to scenarios file for generating training instance and feature vectorizing
        :param train_data: Path to file with train data
        :param transcripts_infile:
        :return:
        """
        self._get_uuid_to_kbs(scenarios)
        # Rudimentary python stop word list
        self.stop_words = set(get_stop_words("en"))
        self._train_tfidf_vectorizer(transcripts_infile)
        inputs, labels = self._process_train_data(train_data)
        self.classifier = self._train(inputs, labels)


    def _train_tfidf_vectorizer(self, data_infile):
        """
        Train a tfidf vectorizer and generate token to tfidf score using provided
        infile
        :param data_infile:
        :return:
        """
        tfidf_vectorizer = TfidfVectorizer()
        transcripts_text = []
        with open(data_infile, "r") as f:
            transcripts = json.load(f)

        for t in transcripts:
            for e in t["events"]:
                if e["action"] == "message":
                    if e["data"] is not None:
                        transcripts_text.append(e["data"])
        # Fit tfidf vectorizer
        tfidf_vectorizer.fit(transcripts_text)
        token_to_tfidf = collections.defaultdict(float)

        # Mapping from token to tf-idf score
        vocab = tfidf_vectorizer.vocabulary_
        weights = tfidf_vectorizer.idf_
        for token, idx in vocab.iteritems():
            token_to_tfidf[token] = weights[idx]

        self.token_to_tfidf = token_to_tfidf


    def _get_uuid_to_kbs(self, scenarios):
        """
        Generate uuid to KB mapping for each scenario
        :return:
        """
        with open(scenarios, "r") as f:
            scenarios_info = json.load(f)

        # Map from uuid to KBs
        uuid_to_kbs = collections.defaultdict(dict)
        for scenario in scenarios_info:
            uuid = scenario["uuid"]
            agent_kbs = {0: set(), 1: set()}
            for agent_idx, kb in enumerate(scenario["kbs"]):
                for item in kb:
                    row_entities = item.items()
                    row_entities = [(e[0], e[1].lower()) for e in row_entities]
                    agent_kbs[agent_idx].update(row_entities)
            uuid_to_kbs[uuid] = agent_kbs

        self.uuid_to_kbs = uuid_to_kbs


    def _feature_func(self, span, entity, agent, uuid):
        """
        Get a series of features between a span of text and a candidate entity
        :param span:
        :param entity:
        :param agent: Id of agent so we know which set of entities to use from scenario
        :param uuid: uuid of scenario to with available KBs
        :return:
        """
        entity_clean = re.sub("-", " ", entity)
        entity_clean_tokens = entity_clean.split()
        span_tokens = span.split()

        try:
            # Set of entities for given agent
            kb_entities = self.uuid_to_kbs[uuid][agent]
            # Only consider entity surface form and not type
            kb_entities = set([e[1] for e in kb_entities])
        except:
            kb_entities = None
            print "No entities found for scenario: {0} and agent: {1}".format(uuid, str(agent))


        features = collections.defaultdict(float)
        if span == entity:
            features["EXACT_MATCH"] = 1.0

        if span in entity:
            features["SUBSTRING"] = 1.0

        ed = editdistance.eval(span, entity)
        if ed == 1:
            features["EDIT_DISTANCE=1"] = 1.0
        elif ed == 2:
            features["EDIT_DISTANCE=2"] = 1.0
        elif ed > 2:
            features["EDIT_DISTANCE>2"] = 1.0

        all_in = True
        for s in span_tokens:
            if s not in entity_clean_tokens:
                all_in = False
        # All tokens of span our a token of the entity
        if all_in:
            features["SUBSET_TOKENS"] = 1.0

        # Edit distance of largest common substring (scaled)
        partial = fuzz.partial_ratio(span, entity)
        if partial >= 90:
            features["PARTIAL_RATIO_>90"] = 1.0
        elif partial >= 50:
            features["PARTIAL_RATIO_>50"] = 1.0
        else:
            features["PARTIAL_RATIO_<50"] = 1.0

        # TF-IDF scores
        span_tfidf = self.token_to_tfidf[span]
        entity_tfidf = self.token_to_tfidf[entity]
        # TODO: Good way to incorporate TF-IDF scores?
        #features["TFIDF_DIFF"] = -1*entity_tfidf + span_tfidf


        # KB context - upweight if entity is in current agent's KB
        if kb_entities is not None and entity in kb_entities:
            features["IN_KB"] = 1.0

        # TODO: Use type features?

        # Feature if both span and entity are stop word
        if span in self.stop_words and entity in self.stop_words:
            features["SPAN_AND_ENTITY_STOP"] = 1.0


        return features


    def _train_featurize(self, inputs):
        """
        Featurize all inputs and labels for training. Different from testing
        because we are computing feature vectors as _feature_func(span, gold_entity) - _feature_fun(span, false_entity)
        :param inputs:
        :return:
        """
        feature_vectors = []
        for input in inputs:
            span = input["span"]
            e1 = input["e1"]
            e2 = input["e2"]
            agent = input["agent"]
            uuid = input["uuid"]
            features_e1 = self._feature_func(span, e1, agent, uuid)
            features_e2 = self._feature_func(span, e2, agent, uuid)

            # Calculate diff between features, represented as dict
            # (Also may want to consider feature concatenation repr.)
            all_features = features_e1.keys() + features_e2.keys()
            feature_diff = {}
            for f in all_features:
                feature_diff[f] = features_e1[f] - features_e2[f]

            feature_vectors.append(feature_diff)

        return feature_vectors


    def _process_train_data(self, train_data):
        """
        Process training data from data file and return as inputs, outputs lists
        :return:
        """
        inputs, labels = [], []
        with open(train_data, "r") as f:
            for line in f:
                _, uuid, agent, span, entity1, entity2, label = line.split("\t")
                inputs.append({"span": span, "e1": entity1, "e2": entity2, "agent": int(agent), "uuid": uuid})
                labels.append(float(label))


        return inputs, labels


    def _train(self, inputs, labels):
        """
        Train on given inputs and labels
        :param inputs: List Dict of (span, entity1, entity2)
        :param labels: List
        :return:
        """
        feature_vecs = self._train_featurize(inputs)
        self.vectorizer = DictVectorizer()
        classifier = LogisticRegression()

        # Feed feature vectorizer to dictvectorizer and then train using logistic regression
        feature_vecs_transform = self.vectorizer.fit_transform(feature_vecs)

        # Train classifier
        classifier.fit(feature_vecs_transform, np.array(labels))

        return classifier


    def score(self, span, entity, agent, uuid):
        """
        Score a span and entity once model is trained
        :param span:
        :param entity:
        :param agent:
        :param uuid:
        :return:
        """
        features = self._feature_func(span, entity, agent, uuid)
        features_transformed = self.vectorizer.transform(features)

        #print "Score: ", self.classifier.predict(features_transformed)
        return self.classifier.predict_proba(features_transformed)


if __name__ == "__main__":
    # TODO: Handle keeping terms like "m.d." intact rather than removing punctuation
    re_pattern = r"[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-data", type=str, help="path to train data")
    parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)
    parser.add_argument("--transcripts", help="transcripts of data collected")

    args = parser.parse_args()

    # TODO: Refactor so that can use ranker for ranking different scenarios from
    # TODO: those used for training!

    ranker = EntityRanker(args.annotated_examples_path, args.scenarios_json, args.ranker_data, args.transcripts)

    print ranker.score("bible", "bible", 1, "S_cxqu6PM56ACAiDLi").squeeze()
