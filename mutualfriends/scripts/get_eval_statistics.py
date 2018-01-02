from argparse import ArgumentParser
from dataset_statistics import *
from src.core.scenario_db import ScenarioDB, add_scenario_arguments
from src.core.schema import Schema
from src.core.util import read_json
from src.model.preprocess import Preprocessor
from src.core.lexicon import Lexicon, add_lexicon_arguments
from collections import defaultdict
from src.core.dataset import Example

parser = ArgumentParser()
add_scenario_arguments(parser)
add_lexicon_arguments(parser)
parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
parser.add_argument('--eval-transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')

parsed_args = parser.parse_args()
schema = Schema(parsed_args.schema_path)
scenario_db = ScenarioDB.from_dict(schema, read_json(parsed_args.scenarios_path))
transcripts = read_json(parsed_args.transcripts)
eval_transcripts = read_json(parsed_args.eval_transcripts)
lexicon = Lexicon(schema, False, scenarios_json=parsed_args.scenarios_path, stop_words=parsed_args.stop_words)
preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical', False)

def compute_statistics(chats):
    speech_act_summary_map = defaultdict(int)
    total = 0.
    for agent, raw in chats:
        ex = Example.from_dict(scenario_db, raw)
        kbs = ex.scenario.kbs
        mentioned_entities = set()
        for i, event in enumerate(ex.events):
            if event.agent == agent:
                if event.action == 'select':
                    utterance = []
                elif event.action == 'message':
                    utterance = preprocessor.process_event(event, kbs[event.agent], mentioned_entities)
                    # Skip empty utterances
                    if not utterance:
                        continue
                    else:
                        utterance = utterance[0]
                        for token in utterance:
                            if is_entity(token):
                                mentioned_entities.add(token[1][0])
                speech_act = get_speech_act(speech_act_summary_map, event, utterance)
                total += 1
    return {'speech_act': {k: speech_act_summary_map[k] / total for k in speech_act_summary_map.keys()}}

uuid_to_chats = {chat['uuid']: chat for chat in transcripts}
scores = eval_transcripts[2]
for question in ['strategic', 'cooperative', 'humanlike']:
    print question
    score_to_chats = defaultdict(list)
    for uuid, d in scores.iteritems():
        chat = uuid_to_chats[uuid]
        for agent in ('0', '1'):
            score = int(d[agent][question][1])  # Median
            score_to_chats[score].append((int(agent), chat))
    sorted_scores = sorted(score_to_chats.keys())
    for score in sorted_scores:
        stats = compute_statistics(score_to_chats[score])
        print score
        speech_act_stats = stats['speech_act']
        for act_type, frac in sorted([(a, b) for a,b in speech_act_stats.items()], key=lambda x:x[1], reverse=True):
            print '%% %s: %2.3f' % (act_type, frac)

