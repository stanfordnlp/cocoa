import argparse
import collections
import editdistance
import json
import numpy as np
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


class EntityRanker(object):
    """
    Learned ranker for ranking candidates of a span of text for the lexicon
    """
    def __init__(self, entity_annotations, scenarios, train_data):
        """
        :param entity_annotations:  Path to JSON of entity annotations
        :param scenarios: Path to scenarios file for generating training instance and feature vectorizing
        :param train_data: Path to file with train data
        :return:
        """
        self._get_uuid_to_kbs(scenarios)
        inputs, labels = self._process_train_data(train_data)
        self.classifier = self._train(inputs, labels)


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

    # TODO: Include KB context
    def _feature_func(self, span, entity):
        """
        Get a series of features between a span of text and a candidate entity
        :param span:
        :param entity:
        :return:
        """
        # Features:
        #   1) Exact match indicator
        #   2) Whether span substring of entity
        #   3) Whether entity contained in KB (need context per training example)
        #   4) Edit distance
        #   5) Whether span is token of entity

        entity_clean = re.sub("-", " ", entity)
        entity_clean_tokens = entity_clean.split()
        span_tokens = span.split()

        features = collections.defaultdict(float)
        if span == entity:
            features["EXACT_MATCH"] = 1.0

        if span in entity:
            features["SUBSTRING"] = 1.0

        ed = editdistance.eval(span, entity)
        features["EDIT_DISTANCE"] = float(ed)

        all_in = True
        for s in span_tokens:
            if s not in entity_clean_tokens:
                all_in = False

        if all_in:
            features["SUBSET_TOKENS"] = 1.0

        return features

    # TODO: also consider original span as candidate

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
            features_e1 = self._feature_func(span, e1)
            features_e2 = self._feature_func(span, e2)

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
                _, span, entity1, entity2, label = line.split("\t")
                inputs.append({"span": span, "e1": entity1, "e2": entity2})
                labels.append(float(label))

        # TODO: Split into train/test data

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


    def score(self, span, entity):
        """
        Score a span and entity once model is trained
        :param span:
        :param entity:
        :return:
        """
        features = self._feature_func(span, entity)
        features_transformed = self.vectorizer.transform(features)

        return self.classifier.predict_proba(features_transformed)


if __name__ == "__main__":
    re_pattern = r"[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-data", type=str, help="path to train data")
    parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)

    args = parser.parse_args()

    ranker = EntityRanker(args.annotated_examples_path, args.scenarios_json, args.ranker_data)

    print ranker.score("penn", "university of pennsylvania")