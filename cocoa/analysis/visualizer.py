import json
from collections import defaultdict
import numpy as np
from itertools import izip, chain
from scipy.stats import ttest_ind as ttest, sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cocoa.core.util import read_json, write_json
from cocoa.core.dataset import Example

from core.scenario import Scenario
from analysis.html_visualizer import HTMLVisualizer

class Visualizer(object):
    # Agents name in the survey
    agents = None
    # Agent names to show in the output
    agent_labels = None

    def __init__(self, chats, surveys=None, worker_ids=None):
        self.chats = []
        for f in chats:
            self.chats.extend(read_json(f))
        self.uuid_to_chat = {chat['uuid']: chat for chat in self.chats}
        if surveys:
            # This is a list because we might have multiple batches of surveys
            self.surveys = [read_json(survey) for survey in surveys]
        if worker_ids:
            self.worker_ids = {}
            for f in worker_ids:
                self.worker_ids.update(read_json(f))
        else:
            self.worker_ids = None

    def worker_stats(self):
        job_counts = defaultdict(int)
        for chat_id, agent_wid in self.worker_ids.iteritems():
            if len(agent_wid) == 2:
                for agent_id, wid in agent_wid.iteritems():
                    # TODO: refactor to is_human
                    if wid != 0:
                        job_counts[wid] += 1
        counts = sorted(job_counts.items(), key=lambda x: x[1], reverse=True)
        for wid, c in counts:
            print wid, c

    def filter(self, *args):
        return None

    @classmethod
    def read_eval(cls, surveys, mask=None):
        '''
        question_scores[question][agent_type] = list of (dialogue_id, agent_id, ratings)
        '''
        question_scores = defaultdict(lambda : defaultdict(list))
        agents = set()
        for survey in surveys:
            cls._read_eval(survey, question_scores, mask=mask, agent_set=agents)
        return list(agents), question_scores

    @classmethod
    def _read_eval(cls, trans, question_scores, mask=None, agent_set=set()):
        dialogue_agents = trans[0]
        dialogue_scores = trans[1]
        dialogue_setting_counts = defaultdict(int)
        for dialogue_id, agent_dict in dialogue_agents.iteritems():
            agent_key = tuple(sorted(agent_dict.values()))
            dialogue_setting_counts[agent_key] += 1
            if mask is not None and not dialogue_id in mask:
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
        print dialogue_setting_counts

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

    def compute_effectiveness_for_system(self, examples, system):
        raise NotImplementedError

    def skip_example(self, ex):
        return False

    def compute_effectiveness(self):
        chats = defaultdict(list)
        for raw in self.chats:
            ex = Example.from_dict(raw, Scenario)
            if self.skip_example(ex):
                continue
            if ex.agents[0] == 'human' and ex.agents[1] == 'human':
                chats['human'].append(ex)
            elif ex.agents[0] != 'human':
                chats[ex.agents[0]].append(ex)
            elif ex.agents[1] != 'human':
                chats[ex.agents[1]].append(ex)

        results = {}
        for system, examples in chats.iteritems():
            if system == 'human':
                continue
            results[system] = self.compute_effectiveness_for_system(examples, system)
            print system, results[system]

    def summarize(self, question_scores=None, summary_stats=('mean',)):
        if not question_scores:
            question_scores = self.question_scores
        summary = defaultdict(lambda : defaultdict(lambda : defaultdict()))
        for summary_stat in summary_stats:
            print '=========== %s ===========' % summary_stat
            for question, agent_scores in question_scores.iteritems():
                if self.question_type(question) == 'str' or question not in self.questions:
                    continue
                results = [(agent, self.summarize_scores(scores, summary_stat), self.get_total(scores)) for agent, scores in agent_scores.iteritems()]
                results = sorted(results, key=lambda x: x[1][0], reverse=True)
                agent_ratings = {}
                for i, (agent, stat, total) in enumerate(results):
                    agent_ratings[agent] = stat[1]
                    summary[question][agent]['score'] = stat[0]
                    summary[question][agent]['sem'] = sem(stat[1]) if len(stat[1]) > 1 else 0
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
                print '============= %s ===============' % self.question_labels[question]
                print '{:<12s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('agent', 'avg_score', 'error', '#score', 'win')
                print '---------------------------------------'
                for i, agent in enumerate(agents):
                    stats = agent_stats[agent]
                    try:
                        print '{:<12s} {:<10.1f} {:<10.2f} {:<10d} {:<10s}'.format(self.agent_labels[agent], stats['score'], stats['sem'], stats['total'], stats['ttest'])
                    except KeyError:
                        continue
        return summary

    def get_dialogue_responses(self, question_scores):
        '''
        Use dialogue_id as key for responses.
        '''
        dialogue_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
        for question, agent_scores in question_scores.iteritems():
            for agent, scores in agent_scores.iteritems():
                for dialogue_id, agent_id, response in scores:
                    try:
                        chat = self.uuid_to_chat[dialogue_id]
                    except KeyError:
                        continue
                    scenario_id = chat['scenario_uuid']
                    dialogue_responses[dialogue_id][agent_id][question] = response
        return dialogue_responses

    def html_visualize(self, viewer_mode, html_output, css_file=None, img_path=None, worker_ids=None):
        chats = []
        scenario_to_chats = defaultdict(set)
        dialogue_responses = None
        if self.question_scores:
            dialogue_responses = self.get_dialogue_responses(self.question_scores)

            # Put chats in the order of responses
            chats_with_survey = set()
            for dialogue_id, agent_responses in dialogue_responses.iteritems():
                chat = self.uuid_to_chat[dialogue_id]
                scenario_id = chat['scenario_uuid']
                chats.append((scenario_id, chat))
                chats_with_survey.add(dialogue_id)
                scenario_to_chats[scenario_id].add(dialogue_id)
            chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]
            # Incomplete chats (redirected, no survey)
            for (dialogue_id, chat) in self.uuid_to_chat.iteritems():
                if dialogue_id not in chats_with_survey:
                    chats.append(chat)
        else:
            for (dialogue_id, chat) in self.uuid_to_chat.iteritems():
                scenario_id = chat['scenario_uuid']
                chats.append((scenario_id, chat))
                scenario_to_chats[scenario_id].add(dialogue_id)
            chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]

        html_visualizer = HTMLVisualizer()
        html_visualizer.visualize(viewer_mode, html_output, chats, responses=dialogue_responses, css_file=css_file, img_path=img_path, worker_ids=worker_ids)
