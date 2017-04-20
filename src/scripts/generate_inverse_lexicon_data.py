import argparse
import json
import re
import sys

sys.path.append("..")
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.basic.schema import Schema
from stop_words import get_stop_words
from src.basic.event import Event
from src.model.vocab import is_entity
from src.model.preprocess import Preprocessor
from src.basic.dataset import Example


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
    parser.add_argument("--output", help="Output path")
    add_lexicon_arguments(parser)

    args = parser.parse_args()

    path = args.schema
    schema = Schema(path)

    re_pattern = r"[\w*\']+|[(\w*&)]+|[\w]+|\.|\(|\)|\\|\"|\/|;|\#|\$|\%|\@|\{|\}|\:"

    lexicon = Lexicon(schema, learned_lex=False, entity_ranker=None, scenarios_json=args.scenarios_json, stop_words=args.stop_words)

    with open(args.annotated_examples_path, "r") as f:
        annotated_examples = json.load(f)

    with open(args.transcripts, "r") as f:
        examples = json.load(f)

    if not args.output:
        fout = open("inverse_lexicon_data.txt", "w")
    else:
        fout = open(args.output, 'w')

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

    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')
    for raw in examples:
        ex = Example.from_dict(None, raw)
        kbs = ex.scenario.kbs
        mentioned_entities = set()
        for i, event in enumerate(ex.events):
            if event.action == 'message':
                utterance = preprocessor.process_event(event, kbs[event.agent], mentioned_entities)
                # Skip empty utterances
                if utterance:
                    utterance = utterance[0]
                    for token in utterance:
                        if is_entity(token):
                            span, entity = token
                            entity, type_ = entity
                            # Entity, Span, Type
                            fout.write(entity + "\t" + span + "\t" + type_ + "\n")

    fout.close()
