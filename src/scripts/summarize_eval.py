import matplotlib
matplotlib.use('Agg')
#font_size = 18
#matplotlib.rcParams.update({k: font_size for k in ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')})
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from src.basic.util import read_json, write_json
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
from itertools import izip
from scipy.stats import ttest_ind as ttest


def my_mode(x):
    return stats.mode(x)[0][0]

def get_dialogue_ids(all_trans):
    ids = []
    for trans in all_trans:
        ids.extend(trans[0].keys())
    return ids

def filter(raw_evals, uuid_to_chats):
    '''
    Only keep scenarios where all 4 agents are evaluated.
    '''
    scenario_to_agents = defaultdict(set)
    scenario_to_chats = defaultdict(set)
    for eval_ in raw_evals:
        dialogue_agents = eval_[0]
        dialogue_scores = eval_[1]
        for dialogue_id, agent_dict in dialogue_agents.iteritems():
            chat = uuid_to_chat[dialogue_id]
            scenario_id = chat['scenario_uuid']
            if isinstance(agent_dict, basestring):
                agent_dict = eval(agent_dict)
            scores = dialogue_scores[dialogue_id]
            for agent_id, results in scores.iteritems():
                if len(results) == 0:
                    continue
                agent_type = agent_dict[str(agent_id)]
                scenario_to_agents[scenario_id].add(agent_type)
                scenario_to_chats[scenario_id].add(dialogue_id)
    good_dialogues = []  # with 4 agent types
    print 'Total scenarios:', len(scenario_to_agents)
    print 'Good scenarios:', sum([1 if len(a) == 4 else 0 for s, a in scenario_to_agents.iteritems()])
    for scenario_id, agents in scenario_to_agents.iteritems():
        if len(agents) == 4:
            good_dialogues.extend(scenario_to_chats[scenario_id])
            #assert len(scenario_to_chats[scenario_id]) >= 4
    filtered_dialogues = set(good_dialogues)
    return filtered_dialogues

def read_eval(trans, question_scores, mask=None):
    dialogue_agents = trans[0]
    dialogue_scores = trans[1]
    for dialogue_id, agent_dict in dialogue_agents.iteritems():
        if mask and not dialogue_id in mask:
            continue
        scores = dialogue_scores[dialogue_id]
        if isinstance(agent_dict, basestring):
            agent_dict = eval(agent_dict)
        for agent_id, results in scores.iteritems():
            agent_type = agent_dict[str(agent_id)]
            for question, ratings in results.iteritems():
                if not isinstance(ratings, list):
                    ratings = (ratings,)
                question_scores[question][agent_type].append((dialogue_id, agent_id, ratings))

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
    ex_scores = [f(ex_scores) for ex_scores in scores]
    return np.mean(ex_scores), ex_scores

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
        if question == 'comments' or question.endswith('text'):
            continue
        for agent, scores in agent_scores.iteritems():
            for dialogue_id, agent_id, response in scores:
                chat = uuid_to_chat[dialogue_id]
                counts = get_stats(chat, int(agent_id), preprocessor)
                for k, v in counts.iteritems():
                    examples[k][question].extend([(agent, v, np.mean(response))])
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

def visualize(viewer_mode, html_output, question_scores, uuid_to_chat):
    dialogue_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for question, agent_scores in question_scores.iteritems():
        for agent, scores in agent_scores.iteritems():
            for dialogue_id, agent_id, response in scores:
                chat = uuid_to_chat[dialogue_id]
                scenario_id = chat['scenario_uuid']
                dialogue_responses[dialogue_id][agent_id][question] = response

    chats = []
    scenario_to_chats = defaultdict(set)
    for dialogue_id, agent_responses in dialogue_responses.iteritems():
        chat = uuid_to_chat[dialogue_id]
        scenario_id = chat['scenario_uuid']
        chats.append((scenario_id, chat))
        scenario_to_chats[scenario_id].add(dialogue_id)
    #for s, c in scenario_to_chats.iteritems():
    #    assert len(c) >= 4
    chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]

    if viewer_mode:
        write_viewer_data(html_output, chats, dialogue_responses)
    else:
        visualize_transcripts(html_output, chats, dialogue_responses)

def one_hist(ax, question, responses_tuples, agents, title, ylabel=False, legend=False):
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.17       # the width of the bars
    i = 0
    all_rects = []
    for (responses, color) in responses_tuples:
        rects = ax.bar(ind + i * width, np.array(responses)*100, width, color=color)
        all_rects.append(rects)
        i += 1

    #ax.set_aspect(0.1, adjustable='box-forced')

    # add some text for labels, title and axes ticks
    ax.set_title(title, fontsize='medium')
    ax.set_xticks(2*width + ind)
    #ax.set_xticklabels(('Bad', 'Mediocre', 'Acceptable', 'Good', 'Excellent'))
    ax.set_xticklabels(('1', '2', '3', '4', '5'))

    if ylabel:
        ax.set_ylabel('Percentage')
    if legend:
        ax.legend([r[0] for r in all_rects], agents, fontsize='medium', loc='best')
    return all_rects

def hist(question_scores, outdir, partner=False):
    question_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    for question, agent_scores in question_scores.iteritems():
        if question == 'comments' or question.endswith('text'):
            continue
        for agent, scores in agent_scores.iteritems():
            for dialogue_id, agent_id, response in scores:
                for r in response:
                    question_responses[question][agent][r - 1] += 1.
    #plt.cla()
    questions = ("fluent", "correct", 'cooperative', "humanlike")
    titles = ('Fluency', 'Correctness', 'Cooperation', 'Human-likeness')
    agents = ('human', 'rulebased', 'static-neural', 'dynamic-neural')
    legends = ('Human', 'Rule', 'StanoNet', 'DynoNet')
    #colors = ["r", "y", "b", "g"]
    colors = ["#33a02c", "#b2df8a", "#1f78b4", "#a6cee3"]
    ncol, nrow = 2, 2
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True)
    for i, (question, ax, title) in enumerate(izip(questions, axes.flat, titles)):
        responses_tuples = []
        for agent, c in izip(agents, colors):
            ratings = question_responses[question][agent]
            ratings = {k: v/sum(ratings.values()) for (k, v) in ratings.items()}
            sorted_ratings = [ratings[k] for k in sorted(ratings.keys())]
            responses_tuples.append((sorted_ratings, c))
        legend = True if (i == 0) else False
        rects = one_hist(ax, question, responses_tuples, legends, title, ylabel=(i==0 or i==2), legend=legend)

    name = 'partner' if partner else 'third'
    axbox = axes.flat[2].get_position()
    plt.tight_layout()
    #plt.savefig('%s/%s_rating.png' % (outdir, name))
    plt.savefig('%s/%s_rating.pdf' % (outdir, name))

def summarize(question_scores):
    summary = defaultdict(lambda : defaultdict(lambda : defaultdict()))
    for summary_stat in ('mean',):
        #print '=========== %s ===========' % summary_stat
        for question, agent_scores in question_scores.iteritems():
            if question == 'comments' or question.endswith('text'):
                continue
            results = [(agent, summarize_scores(scores, summary_stat), get_total(scores)) for agent, scores in agent_scores.iteritems()]
            results = sorted(results, key=lambda x: x[1][0], reverse=True)
            agent_ratings = {}
            for i, (agent, stat, total) in enumerate(results):
                agent_ratings[agent] = stat[1]
                summary[question][agent]['score'] = stat[0]
                summary[question][agent]['total'] = total
                summary[question][agent]['ttest'] = ''
            # T-test
            agents = ('human', 'rulebased', 'static-neural', 'dynamic-neural')
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    result = ttest(agent_ratings[agents[i]], agent_ratings[agents[j]])
                    #print agents[i], agents[j], result
                    t, p = result
                    if p < 0.05:
                        if t > 0:
                            win_agent, lose_agent = agents[i], agents[j]
                        else:
                            win_agent, lose_agent = agents[j], agents[i]
                        summary[question][win_agent]['ttest'] += lose_agent[0]
        # Print
        agent_labels = ('Human', 'Rule-based', 'StanoNet', 'DynoNet')
        for question, agent_stats in summary.iteritems():
            print '============= %s ===============' % question.upper()
            print '{:<12s} {:<10s} {:<10s} {:<10s}'.format('agent', 'avg_score', '#score', 'win')
            print '---------------------------------------'
            for i, agent in enumerate(agents):
                stats = agent_stats[agent]
                print '{:<12s} {:<10.1f} {:<10d} {:<10s}'.format(agent_labels[i], stats['score'], stats['total'], stats['ttest'])
    return summary

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval-transcripts', nargs='+', help='Path to directory containing evaluation transcripts')
    parser.add_argument('--dialogue-transcripts', help='Path to directory containing dialogue transcripts')
    parser.add_argument('--analyze', default=False, action='store_true', help='Analyze human ratings')
    parser.add_argument('--summary', default=False, action='store_true', help='Summarize human ratings')
    parser.add_argument('--hist', default=False, action='store_true', help='Plot histgram of ratings')
    parser.add_argument('--visualize', action='store_true', help='Output html files')
    parser.add_argument('--outdir', default='.', help='Output dir')
    parser.add_argument('--stats', default='stats.json', help='Path to stats file')
    parser.add_argument('--partner', default=False, action='store_true', help='Whether this is from partner survey')
    add_scenario_arguments(parser)
    add_lexicon_arguments(parser)
    add_visualization_arguments(parser)
    args = parser.parse_args()

    raw_eval = [read_json(trans) for trans in args.eval_transcripts]
    question_scores = defaultdict(lambda : defaultdict(list))
    raw_chats = read_json(args.dialogue_transcripts)
    uuid_to_chat = {chat['uuid']: chat for chat in raw_chats}
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dialogue_ids = filter(raw_eval, uuid_to_chat)

    for eval_ in raw_eval:
        read_eval(eval_, question_scores, mask=dialogue_ids)

    if args.hist:
        hist(question_scores, args.outdir, partner=args.partner)

    if args.summary:
        summary = summarize(question_scores)
        write_json(summary, args.stats)

    if args.analyze:
        schema = Schema(args.schema_path)
        lexicon = Lexicon(schema, False, scenarios_json=args.scenarios_path, stop_words=args.stop_words)
        preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')
        analyze(question_scores, uuid_to_chat, preprocessor)

    # Visualize
    if args.html_output:
        visualize(args.viewer_mode, args.html_output, question_scores, uuid_to_chat)

