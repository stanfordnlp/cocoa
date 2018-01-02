import argparse
import json
import re
import sys

sys.path.append("..")
from basic.lexicon import Lexicon
from basic.schema import Schema
from stop_words import get_stop_words


"""
Generates data file for training entity ranker
File format:
    Idx \t span \t entity_1 \t entity_2 \t label

where label is 1 if entity_1 is true entity for span and 0 otherwise
"""

parser = argparse.ArgumentParser()
parser.add_argument("--schema-path", help="Path to schema file governs scenarios",
                    type=str)
parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
parser.add_argument("--scenarios-json", help="Json of scenarios", type=str)

args = parser.parse_args()


fout = open("../../output/entity_ranker_data.txt", "w")

re_pattern = r"[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"
schema = Schema(args.schema_path)
lexicon = Lexicon(schema)

with open(args.annotated_examples_path, "r") as f:
    annotated_examples = json.load(f)

# Stop words
sw = set(get_stop_words("en"))

# Iterate over gold annotations, get candidate set, and form training instances
# TODO: May want to consider adding more examples by finding candidates over all spans of utterances
idx = 0
for ex in annotated_examples:
    uuid = ex["scenario_uuid"]
    for e in ex["events"]:
        msg_data = e["data"]
        action = e["action"]
        agent = e["agent"]
        if msg_data is not None and isinstance(msg_data, unicode):
            raw_msg_tokens = re.findall(re_pattern, msg_data.lower())

            msg_stop_words = [rmt for rmt in raw_msg_tokens if rmt in sw]

        if action == "message":
            for a in e["entityAnnotation"]:
                span = re.sub("-|\.", " ", a["span"].lower()).strip()
                gold_entity = a["entity"].lower()

                raw_tokens = re.findall(re_pattern, span.lower())
                #lower_raw_tokens = [r.lower() for r in raw_tokens]
                linked, _ = lexicon.link_entity(raw_tokens, return_entities=True, kb_entities=None)
                # Add train example for every candidate entry
                for l in linked:
                    if isinstance(l, list):
                        for candidate_entity in l:
                            if candidate_entity[0] != gold_entity:
                                # Record scenario UUID + agent idx
                                fout.write(str(idx) + "\t" + uuid + "\t" + str(agent) + "\t" + span + "\t" + gold_entity + "\t" + candidate_entity[0] + "\t" + "1" + "\n")
                                idx += 1
                                fout.write(str(idx) + "\t" + uuid + "\t" + str(agent) + "\t" + span + "\t" + candidate_entity[0] + "\t" + gold_entity + "\t" + "0" + "\n")
                                idx += 1

                                # Make training examples for stop words
                                for msw in msg_stop_words:
                                    fout.write(str(idx) + "\t" + uuid + "\t" + str(agent) + "\t" + msw + "\t" + msw + "\t" + candidate_entity[0] + "\t" + "1" + "\n")
                                    idx += 1
                                    fout.write(str(idx) + "\t" + uuid + "\t" + str(agent) + "\t" + msw + "\t" + candidate_entity[0] + "\t" + msw + "\t" + "0" + "\n")
                                    idx += 1



fout.close()
