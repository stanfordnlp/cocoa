import argparse
import json
import re
import sys

sys.path.append("..")
from basic.lexicon import Lexicon
from basic.schema import Schema


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
args = parser.parse_args()


fout = open("entity_ranker_data.txt", "w")

re_pattern = r"[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"
schema = Schema(args.schema_path)
lexicon = Lexicon(schema)

with open(args.annotated_examples_path, "r") as f:
    annotated_examples = json.load(f)

# Iterate over gold annotations, get candidate set, and form training instances
# TODO: May want to consider adding more examples by finding candidates over all spans of utterances
idx = 0
for ex in annotated_examples:
    for e in ex["events"]:
        msg_data = e["data"]
        action = e["action"]
        if action == "message":
            for a in e["entityAnnotation"]:
                span = re.sub("-|\.", " ", a["span"].lower()).strip()
                gold_entity = a["entity"].lower()

                raw_tokens = re.findall(re_pattern, span.lower())
                lower_raw_tokens = [r.lower() for r in raw_tokens]
                linked, _ = lexicon.link_entity(lower_raw_tokens, return_entities=True, kb_entities=None)
                for l in linked:
                    if isinstance(l, list):
                        for candidate_entity in l:
                            if candidate_entity[0] != gold_entity:
                                fout.write(str(idx) + "\t" + span + "\t" + gold_entity + "\t" + candidate_entity[0] + "\t" + "1" + "\n")
                                idx += 1
                                fout.write(str(idx) + "\t" + span + "\t" + candidate_entity[0] + "\t" + gold_entity + "\t" + "0" + "\n")
                                idx += 1


fout.close()
