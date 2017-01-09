import argparse
import collections
import json
import os
import re
import sys
import time

# Hack to be able to import modules one directory up
sys.path.append("..")
from basic.entity_ranker import EntityRanker
from basic.lexicon import Lexicon
from basic.schema import Schema

"""
Runs lexicon on transcripts of MTurk conversations and entity annotated dataset
"""

parser = argparse.ArgumentParser()
parser.add_argument("--schema", help="Path to schema file governs scenarios",
                    type=str)
parser.add_argument("--transcripts", help="Json of all examples", type=str)
parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)
parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
parser.add_argument("--ranker-data", type=str, help="path to train data")

args = parser.parse_args()


def compute_f1(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)

    return 2. * precision * recall / (precision + recall), precision, recall


def get_tp_fp_fn(gold_annotation, candidate_annotation):
    """
    Get the true positive, false positive, and false negative for the sets
    :param gold_annotation:
    :param candidate_annotation:
    :return:
    """
    tp, fp, fn = 0., 0., 0.
    for c in candidate_annotation:
        if c in gold_annotation:
            tp += 1
        else:
            fp += 1

    for g in gold_annotation:
        if g not in candidate_annotation:
            fn += 1

    # How to handle empty gold and candidate sets?

    return {"true_positive": tp, "false_positive": fp, "false_negative": fn}


def entity_link_examples_file(lexicon, examples_infile, processed_outfile, re_pattern):
    """
    Processes examples in given file, entity linking with the provided lexicon
    :param lexicon: Lexicon being used for linking
    :param examples_infile: Path to file with examples
    :return:
    """
    with open(examples_infile, "r") as f:
        examples = json.load(f)

    outfile = open(processed_outfile, "w")

    for ex in examples:
        events = ex["events"]

        num_sentences = 0
        for e in events:
            agent = e["agent"]
            msg_data = e["data"]
            action = e["action"]
            if action == "message":
                if msg_data is not None:
                    num_sentences += 1
                    raw_tokens = re.findall(re_pattern, msg_data)
                    lower_raw_tokens = [r.lower() for r in raw_tokens]
                    outfile.write("Agent {0}: ".format(agent) + msg_data + "\n")
                    linked = lexicon.link_entity(lower_raw_tokens, return_entities=True)

                    outfile.write("Linked: " + str(linked) + "\n\n")
                    outfile.write("-"*10 + "\n")

    outfile.close()


def eval_lexicon(lexicon, examples, re_pattern):
    """
    Evaluate lexicon given list of examples
    :param lexicon:
    :param examples:
    :param re_pattern:
    :return:
    """
    total_num_annotations = 0
    total_num_sentences = 0
    total_tp, total_fp, total_fn = 0., 0., 0.
    for ex in examples:
        scenario_uuid = ex["scenario_uuid"]

        for e in ex["events"]:
            msg_data = e["data"]
            action = e["action"]
            agent = e["agent"]

            if action == "message":
                total_num_sentences += 1

                gold_annotation = []
                for a in e["entityAnnotation"]:
                    span = re.sub("-|\.", " ", a["span"].lower()).strip()
                    entity = a["entity"].lower()
                    gold_annotation.append((span, entity))

                raw_tokens = re.findall(re_pattern, msg_data)
                lower_raw_tokens = [r.lower() for r in raw_tokens]
                _, candidate_annotation = lexicon.link_entity(lower_raw_tokens, return_entities=True, agent=agent, uuid=scenario_uuid)

                # Modify each candidate from (surface form, (canonical, type)) -> (surface form, canonical) for comparison
                candidate_annotation = [(c[0], c[1][0]) for c in candidate_annotation]

                total_num_annotations += len(gold_annotation)
                metrics = get_tp_fp_fn(gold_annotation, candidate_annotation)
                tp = metrics["true_positive"]
                fp = metrics["false_positive"]
                fn = metrics["false_negative"]
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Output mistakes to stdout
                if fp >= 1 or fn >= 1:
                    print msg_data
                    print "gold: ", gold_annotation
                    print "candidate: ", candidate_annotation
                    print "TP: {0}, FP: {1}, FN: {2}".format(tp, fp, fn)
                    print "-"*10

    avg_f1, avg_precision, avg_recall = compute_f1(total_tp, total_fp, total_fn)
    print "Avg f1 over {0} annotations: {1}, {2}, {3}".format(total_num_annotations,
                                                              avg_f1, avg_precision, avg_recall)

    return avg_f1


if __name__ == "__main__":
    # Regex to remove all punctuation in utterances
    # TODO: Use easier regex
    re_pattern = r"[\w*\']+|[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"
    schema = Schema(args.schema)

    start = time.time()

    ranker = EntityRanker(args.annotated_examples_path, args.scenarios_json, args.ranker_data, args.transcripts)

    with open(args.annotated_examples_path, "r") as f:
        examples = json.load(f)


    output_dir = os.path.dirname(os.path.dirname(os.getcwd())) + "/output"
    lexicon = Lexicon(schema, learned_lex=False, entity_ranker=ranker, scenarios_json=args.scenarios_json)

    eval_lexicon(lexicon, examples, re_pattern)
    print "Total time: ", time.time() - start
