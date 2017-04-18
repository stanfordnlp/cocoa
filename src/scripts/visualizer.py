'''
Visualize evaluation results.
'''

import src.config as config
from src.basic.util import read_json, write_json
from html_visualizer import HTMLVisualizer, add_html_visualizer_arguments
from collections import defaultdict
import numpy as np
from itertools import izip
from scipy.stats import ttest_ind as ttest
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

class Visualizer(object):
    '''
    Factory of visualizers that summarize and render surveys and transcripts.
    '''
    @staticmethod
    def get_visualizer(*args):
        if config.task == config.MutualFriends:
            return MutualFriendsVisualizer(*args)
        elif config.task == config.Negotiation:
            return NegotiationVisualizer(*args)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class BaseVisualizer(object):
    # Agents name in the survey
    agents = None
    # Agent names to show in the output
    agent_labels = None

    def __init__(self, chats, surveys=None):
        self.chats = read_json(chats)
        self.uuid_to_chat = {chat['uuid']: chat for chat in self.chats}
        if surveys:
            # This is a list because we might have multiple batches of surveys
            self.surveys = [read_json(survey) for survey in surveys]
            mask = self.filter(self.surveys)
            self.question_scores = self.read_eval(self.surveys, mask)

    def filter(self, evals):
        return None

    def read_eval(self, surveys, mask=None):
        question_scores = defaultdict(lambda : defaultdict(list))
        agents = set()
        for survey in surveys:
            self._read_eval(survey, question_scores, mask=mask, agent_set=agents)
        self.agents = list(agents)
        return question_scores

    def _read_eval(self, trans, question_scores, mask=None, agent_set=set()):
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
                agent_set.add(agent_type)
                for question, ratings in results.iteritems():
                    if not isinstance(ratings, list):
                        ratings = (ratings,)
                    ratings = [3 if x == 'null' else x for x in ratings]
                    question_scores[question][agent_type].append((dialogue_id, agent_id, ratings))

    def question_type(self, question):
        '''
        Type of a survey question, e.g. string or numerical.
        '''
        raise NotImplementedError

    def one_hist(self, ax, question, responses_tuples, agents, title, ylabel=False, legend=False):
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

    def hist(self, outdir, question_scores=None, partner=False):
        if not question_scores:
            question_scores = self.question_scores
        question_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        for question, agent_scores in question_scores.iteritems():
            if self.question_type(question) == 'str':
                continue
            for agent, scores in agent_scores.iteritems():
                for dialogue_id, agent_id, response in scores:
                    for r in response:
                        question_responses[question][agent][r - 1] += 1.
        questions = self.questions
        titles = ('Fluency', 'Correctness', 'Cooperation', 'Human-likeness')
        agents = self.agents
        legends = [self.agent_labels[a] for a in agents]
        colors = self.colors
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
            rects = self.one_hist(ax, question, responses_tuples, legends, title, ylabel=(i==0 or i==2), legend=legend)

        name = 'partner' if partner else 'third'
        axbox = axes.flat[2].get_position()
        plt.tight_layout()
        #plt.savefig('%s/%s_rating.png' % (outdir, name))
        plt.savefig('%s/%s_rating.pdf' % (outdir, name))

    @classmethod
    def summarize_scores(cls, scores, stat):
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

    @classmethod
    def get_total(cls, scores):
        scores = [x[2] for x in scores]
        return sum([len(x) for x in scores])

    def summarize(self, question_scores=None, summary_stats=('mean',)):
        if not question_scores:
            question_scores = self.question_scores
        summary = defaultdict(lambda : defaultdict(lambda : defaultdict()))
        for summary_stat in summary_stats:
            print '=========== %s ===========' % summary_stat
            for question, agent_scores in question_scores.iteritems():
                if self.question_type(question) == 'str':
                    continue
                results = [(agent, self.summarize_scores(scores, summary_stat), self.get_total(scores)) for agent, scores in agent_scores.iteritems()]
                results = sorted(results, key=lambda x: x[1][0], reverse=True)
                agent_ratings = {}
                for i, (agent, stat, total) in enumerate(results):
                    agent_ratings[agent] = stat[1]
                    summary[question][agent]['score'] = stat[0]
                    summary[question][agent]['total'] = total
                    summary[question][agent]['ttest'] = ''
                # T-test
                agents = self.agents
                for i in range(len(agents)):
                    for j in range(i+1, len(agents)):
                        try:
                            result = ttest(agent_ratings[agents[i]], agent_ratings[agents[j]])
                        except KeyError:
                            continue
                        #print agents[i], agents[j], result
                        t, p = result
                        if p < 0.05:
                            if t > 0:
                                win_agent, lose_agent = agents[i], agents[j]
                            else:
                                win_agent, lose_agent = agents[j], agents[i]
                            summary[question][win_agent]['ttest'] += lose_agent[0]
            # Print
            for question, agent_stats in summary.iteritems():
                print '============= %s ===============' % question.upper()
                print '{:<12s} {:<10s} {:<10s} {:<10s}'.format('agent', 'avg_score', '#score', 'win')
                print '---------------------------------------'
                for i, agent in enumerate(agents):
                    stats = agent_stats[agent]
                    print '{:<12s} {:<10.1f} {:<10d} {:<10s}'.format(self.agent_labels[agent], stats['score'], stats['total'], stats['ttest'])
        return summary

    def get_dialogue_responses(self, question_scores):
        '''
        Use dialogue_id as key for responses.
        '''
        dialogue_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
        for question, agent_scores in question_scores.iteritems():
            for agent, scores in agent_scores.iteritems():
                for dialogue_id, agent_id, response in scores:
                    chat = self.uuid_to_chat[dialogue_id]
                    scenario_id = chat['scenario_uuid']
                    dialogue_responses[dialogue_id][agent_id][question] = response
        return dialogue_responses

    def html_visualize(self, viewer_mode, html_output, css_file=None):
        dialogue_responses = self.get_dialogue_responses(self.question_scores)

        # Put chats in the order of responses
        chats = []
        scenario_to_chats = defaultdict(set)
        for dialogue_id, agent_responses in dialogue_responses.iteritems():
            chat = self.uuid_to_chat[dialogue_id]
            scenario_id = chat['scenario_uuid']
            chats.append((scenario_id, chat))
            scenario_to_chats[scenario_id].add(dialogue_id)
        chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]

        html_visualizer = HTMLVisualizer.get_html_visualizer()
        html_visualizer.visualize(viewer_mode, html_output, chats, responses=dialogue_responses, css_file=css_file)

class NegotiationVisualizer(BaseVisualizer):
    agents = ('human', 'rulebased')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based'}
    questions = ('fluent', 'honest', 'persuasive', 'fair')
    question_labels = {"fluent": 'Fluency', "honest": 'Honesty', 'persuasive': 'Persuasiveness', "fair": 'Fairness'}

    def question_type(self, question):
        if question == 'comments':
            return 'str'
        else:
            return 'num'


class MutualFriendsVisualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'static-neural', 'dynamic-neural')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'static-neural': 'StanoNet', 'dynamic-neural': 'DynoNet'}
    questions = ("fluent", "correct", 'cooperative', "humanlike")
    question_labels = {"fluent": 'Fluency', "correct": 'Correctness', 'cooperative': 'Cooperation', "humanlike": 'Human-likeness'}
    colors = ("#33a02c", "#b2df8a", "#1f78b4", "#a6cee3")

    def question_type(self, question):
        if question == 'comments' or question.endswith('text'):
            return 'str'
        else:
            return 'num'

    def filter(self, raw_evals):
        '''
        Only keep scenarios where all 4 agents are evaluated.
        '''
        scenario_to_agents = defaultdict(set)
        scenario_to_chats = defaultdict(set)
        for eval_ in raw_evals:
            dialogue_agents = eval_[0]
            dialogue_scores = eval_[1]
            for dialogue_id, agent_dict in dialogue_agents.iteritems():
                chat = self.uuid_to_chat[dialogue_id]
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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--survey-transcripts', nargs='+', help='Path to directory containing evaluation transcripts')
    parser.add_argument('--dialogue-transcripts', help='Path to directory containing dialogue transcripts')
    parser.add_argument('--summary', default=False, action='store_true', help='Summarize human ratings')
    parser.add_argument('--hist', default=False, action='store_true', help='Plot histgram of ratings')
    parser.add_argument('--html-visualize', action='store_true', help='Output html files')
    parser.add_argument('--outdir', default='.', help='Output dir')
    parser.add_argument('--stats', default='stats.json', help='Path to stats file')
    parser.add_argument('--partner', default=False, action='store_true', help='Whether this is from partner survey')
    add_html_visualizer_arguments(parser)
    args = parser.parse_args()

    visualizer = Visualizer.get_visualizer(args.dialogue_transcripts, args.survey_transcripts)

    if args.hist:
        visualizer.hist(question_scores, args.outdir, partner=args.partner)

    if args.summary:
        summary = visualizer.summarize()
        write_json(summary, args.stats)

    if args.html_output:
        visualizer.html_visualize(args.viewer_mode, args.html_output, css_file=args.css_file)

