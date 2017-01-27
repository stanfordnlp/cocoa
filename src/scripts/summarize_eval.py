import matplotlib
matplotlib.use('Agg')
font_size = 18
matplotlib.rcParams.update({k: font_size for k in ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')})
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from src.basic.util import read_json
from src.scripts.visualize_data import *
from dataset_statistics import *
from src.model.preprocess import Preprocessor
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.dataset import Example
from src.basic.schema import Schema
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.model.vocab import is_entity
from collections import defaultdict
import numpy as np
from scipy import stats
from src.lib import logstats

def my_mode(x):
    return stats.mode(x)[0][0]

def read_eval(trans, question_scores):
    dialogue_agents = trans[0]
    dialogue_scores = trans[1]
    for dialogue_id, agent_dict in dialogue_agents.iteritems():
        scores = dialogue_scores[dialogue_id]
        if isinstance(agent_dict, basestring):
            agent_dict = eval(agent_dict)
        for agent_id, results in scores.iteritems():
            agent_type = agent_dict[str(agent_id)]
            for question, ratings in results.iteritems():
                question_scores[question][agent_type].append((dialogue_id, agent_id, ratings))
        #for question, ratings in scores.iteritems():
        #    agent_types = dialogue_agents[dialogue_id]
        #    if agent_types['0'] == 'human':
        #        agent_type = agent_types['1']
        #    else:
        #        agent_type = agent_types['0']
        #    if not isinstance(ratings, list):
        #        ratings = (ratings,)
        #    question_scores[question][agent_type].append((dialogue_id, ratings))

def summarize_scores(scores, stat):
    scores = [x[2] for x in scores]
    if stat == 'median':
        f = np.median
    elif stat == 'mean':
        f = np.mean
    elif stat == 'mode':
        f = my_mode
    else:
        raise ValueError('Unknown stats')
    return np.mean([f(ex_scores) for ex_scores in scores])

def get_total(scores):
    scores = [x[2] for x in scores]
    return sum([len(x) for x in scores])

def get_stats(chat, agent_id, preprocessor):
    ex = Example.from_dict(None, chat)
    kbs = ex.scenario.kbs
    mentioned_entities = set()
    stats = {}
    vocab = set()
    for i, event in enumerate(ex.events):
        if agent_id != event.agent:
            continue
        if event.action == 'select':
            utterance = []
            logstats.update_summary_map(stats, {'num_select': 1})
        elif event.action == 'message':
            utterance = preprocessor.process_event(event, kbs[event.agent], mentioned_entities)
            # Skip empty utterances
            if not utterance:
                continue
            else:
                utterance = utterance[0]
                for token in utterance:
                    if is_entity(token):
                        logstats.update_summary_map(stats, {'num_entity': 1})
                        mentioned_entities.add(token[1][0])
                    else:
                        vocab.add(token)
                logstats.update_summary_map(stats, {'utterance_len': len(utterance)})
        speech_act = get_speech_act(defaultdict(int), event, utterance)
        if speech_act[0] in ('inform', 'ask', 'answer'):
            logstats.update_summary_map(stats, {'SA_'+speech_act[0]: 1})
        logstats.update_summary_map(stats, {'num_utterance': 1})

    new_stats = {}
    for k in stats:
        if k in ('num_select', 'num_utterance', 'num_entity'):
            new_stats[k] = stats[k]['sum']
        elif k in ('utterance_len',):
            new_stats[k] = stats[k]['mean']
        elif k.startswith('SA_'):
            new_stats[k] = stats[k]['sum']
    new_stats['vocab_size'] = len(vocab)
    return new_stats

def scatter_plot(question, stat_name, agents, counts, scores):
    assert len(agents) == len(counts) and len(counts) == len(scores)
    color_map = {'human': 'r', 'rulebased': 'g', 'dynamic-neural': 'c', 'static-neural': 'b'}
    colors = [color_map[a] for a in agents]
    plt.cla()
    plt.scatter(counts, scores, c=colors, alpha=0.5)
    plt.xlabel(stat_name)
    plt.ylabel('%s scores' % question)
    plt.tight_layout()
    plt.savefig('%s_%s.png' % (question, stat_name))

def analyze(question_scores, uuid_to_chat, preprocessor):
    # factor -> question -> (agent_type, scores)
    examples = defaultdict(lambda : defaultdict(list))
    for question, agent_scores in question_scores.iteritems():
        for agent, scores in agent_scores.iteritems():
            for dialogue_id, agent_id, response in scores:
                chat = uuid_to_chat[dialogue_id]
                counts = get_stats(chat, int(agent_id), preprocessor)
                for k, v in counts.iteritems():
                    #examples[k][question].extend([(agent, v, s) for s in response])
                    examples[k][question].extend([(agent, v, my_mode(response))])
    # plot
    corr = []
    for stat_name, question_scores in examples.iteritems():
        for question, scores in question_scores.iteritems():
            agents = [x[0] for x in scores]
            counts = [x[1] for x in scores]
            ratings = [x[2] for x in scores]
            corr.append((question, stat_name, stats.pearsonr(counts, ratings)))
            #scatter_plot(question, stat_name, agents, counts, ratings)
    corr = sorted(corr, key=lambda x: abs(x[2][0]))
    for x in corr:
        print x

def visualize(html_output, question_scores, uuid_to_chat):
    dialogue_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for question, agent_scores in question_scores.iteritems():
        for agent, scores in agent_scores.iteritems():
            for dialogue_id, _, response in scores:
                dialogue_responses[dialogue_id][agent][question] = response

    responses = []
    chats = []
    for dialogue_id, agent_responses in dialogue_responses.iteritems():
        for agent, response in agent_responses.iteritems():
            chat = uuid_to_chat[dialogue_id]
            scenario_id = chat['scenario_uuid']
            chats.append((scenario_id, agent, chat))
            responses.append((scenario_id, agent, response))
    chats = [x[2] for x in sorted(chats, key=lambda x: (x[0], x[1]))]
    responses = [x[1:] for x in sorted(responses, key=lambda x: (x[0], x[1]))]

    visualize_transcripts(html_output, scenario_db, chats, dialogue_responses)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval-transcripts', nargs='+', help='Path to directory containing evaluation transcripts')
    parser.add_argument('--dialogue-transcripts', help='Path to directory containing dialogue transcripts')
    parser.add_argument('--html-output', default=None, help='Path to HTML output')
    parser.add_argument('--analyze', default=False, action='store_true', help='Analyze human ratings')
    parser.add_argument('--summary', default=False, action='store_true', help='Summarize human ratings')
    add_scenario_arguments(parser)
    add_lexicon_arguments(parser)
    args = parser.parse_args()

    raw_eval = [read_json(trans) for trans in args.eval_transcripts]
    question_scores = defaultdict(lambda : defaultdict(list))
    raw_chats = read_json(args.dialogue_transcripts)
    uuid_to_chat = {chat['uuid']: chat for chat in raw_chats}
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))

    for eval_ in raw_eval:
        read_eval(eval_, question_scores)

    # Median of median
    if args.summary:
        #for summary_stat in ('median', 'mean', 'mode'):
        for summary_stat in ('mode',):
            print '=========== %s ===========' % summary_stat
            for question, agent_scores in question_scores.iteritems():
                if question == 'comments':
                    continue
                print '------------- %s ---------------' % question
                results = [(agent, summarize_scores(scores, summary_stat), get_total(scores)) for agent, scores in agent_scores.iteritems()]
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for agent, stat, total in results:
                    print '{:<15s} {:<10.1f} {:<10d}'.format(agent, stat, total)

    if args.analyze:
        schema = Schema(args.schema_path)
        lexicon = Lexicon(schema, False, scenarios_json=args.scenarios_path, stop_words=args.stop_words)
        preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical', False)
        analyze(question_scores, uuid_to_chat, preprocessor)

    # Visualize
    if args.html_output is not None:
        if not os.path.exists(os.path.dirname(args.html_output)) and len(os.path.dirname(args.html_output)) > 0:
            os.makedirs(os.path.dirname(args.html_output))
        visualize(args.html_output, question_scores, uuid_to_chat)

