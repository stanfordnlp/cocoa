import argparse
import json
import re
import sys

sys.path.append("..")
from basic.lexicon import Lexicon
from basic.schema import Schema
from stop_words import get_stop_words

"""
Generate data for building inverse lexicon by
running regular lexicon on transcripts. Data outputted
should be of form:

    <entity \t <span> \t <type>

for each entity linked by lexicon
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("arguments for basic testing lexicon")
    parser.add_argument("--schema", type=str, help="path to schema to use")
    parser.add_argument("--ranker-data", type=str, help="path to train data")
    parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)
    parser.add_argument("--transcripts", help="Json file of all transcripts collected")

    args = parser.parse_args()

    path = args.schema
    schema = Schema(path)

    re_pattern = r"[\w*\']+|[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"

    lexicon = Lexicon(schema, learned_lex=False, entity_ranker=None, scenarios_json=args.scenarios_json)

    with open(args.annotated_examples_path, "r") as f:
        annotated_examples = json.load(f)

    with open(args.transcripts, "r") as f:
        examples = json.load(f)

    fout = open("../../data/inverse_lexicon_data.txt", "w")

    # Process annotated examples
    for ex in annotated_examples:
        scenario_uuid = ex["scenario_uuid"]

        for e in ex["events"]:
            msg_data = e["data"]
            action = e["action"]
            agent = e["agent"]

            if action == "message":
                raw_tokens = re.findall(re_pattern, msg_data)
                lower_raw_tokens = [r.lower() for r in raw_tokens]
                _, candidate_annotation = lexicon.link_entity(lower_raw_tokens, return_entities=True, agent=agent, uuid=scenario_uuid)

                for c in candidate_annotation:
                    # Entity, Span, Type
                    fout.write(c[1][0] + "\t" + c[0] + "\t" + c[1][1] + "\n")

    # TODO: Process transcripts of all conversations

    fout.close()
