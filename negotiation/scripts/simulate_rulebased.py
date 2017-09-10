import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand
from scipy.stats import sem

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB, add_scenario_arguments

from core.scenario import Scenario
from core.controller import Controller
from systems import add_system_arguments, get_system
from sessions.rulebased_session import random_configs, default_config, Config
from analysis.analyze_strategy import StrategyAnalyzer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
    parser.add_argument('--agent-id', default=0, type=int, help='Which agent to try')
    add_scenario_arguments(parser)
    add_system_arguments(parser)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
    agents = [get_system('rulebased', args, schema) for _ in (0, 1)]

    configs = random_configs(10)
    scenarios = scenario_db.scenarios_list[:20]
    #scenarios = [scenarios[0] for _ in xrange(10)]
    max_turns = args.max_turns
    success_rates = []
    mean_margins = []
    margin_stes = []

    def eval_bot(params):
        success = []
        margins = []
        agent_id = args.agent_id
        baseline_agent_id = 1 - agent_id
        config = Config(**params)
        for scenario in scenarios:
            baseline_agent = agents[baseline_agent_id].new_session(baseline_agent_id, scenario.kbs[baseline_agent_id], default_config)
            agent = agents[agent_id].new_session(agent_id, scenario.kbs[agent_id], config)
            controller = Controller(scenario, [baseline_agent, agent])
            ex = controller.simulate(max_turns, verbose=False)
            #import sys; sys.exit()

            if StrategyAnalyzer.has_deal(ex):
                success.append(1)
                final_price = ex.outcome['offer']['price']
                margin = StrategyAnalyzer.get_margin(ex, final_price, 1, scenario.kbs[agent_id].facts['personal']['Role'], remove_outlier=False)
                margins.append(margin)
            else:
                success.append(0)
        return {
                'loss': -1. * np.mean(margins) if len(margins) > 0 else 1000,
                'status': STATUS_OK,
                'ste': sem(margins),
                'success_rate': np.mean(success),
                }

    params = {
            'overshoot': .2,
            'bottomline_fraction': .3,
            'compromise_fraction': .75,
            'good_deal_threshold': .88,
            }
    #eval_bot(params)

    space = {
            'overshoot': hp.uniform('overshoot', 0., .5),
            'bottomline_fraction': hp.uniform('bottomline_fraction', .1, .8),
            'compromise_fraction': hp.uniform('compromise_fraction', .1, .8),
            'good_deal_threshold': hp.uniform('good_deal_threshold', .1, .8),
            }

    trials = Trials()
    best = fmin(eval_bot, space, algo=tpe.suggest, max_evals=100, trials=trials)
    tpe_losses = trials.losses()
    trials = Trials()
    best = fmin(eval_bot, space, algo=rand.suggest, max_evals=100, trials=trials)
    rand_losses = trials.losses()
    fig, ax = plt.subplots()
    x = range(100)
    ax.plot(x, tpe_losses, label='TPE')
    ax.plot(x, rand_losses, label='random search')
    plt.legend()
    plt.savefig('opt.png')
    import sys; sys.exit()


    best_trial = trials.trials[np.argmin(trials.losses())]
    print 'best config:', best
    print "best margin:", -best_trial['result']['loss']
    print 'best success rate:', best_trial['result']['success_rate']
    print 'best ste:', best_trial['result']['ste']

    # Plot
    parameters = space.keys()
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15,5))
    cmap = plt.cm.jet
    valid_trials = [t for t in trials.trials if not t['result']['loss'] == 1000]
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in valid_trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
        axes[i].set_title(val)
    plt.savefig('params-{agent}.png'.format(agent=args.agent_id))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
    y = [-t['result']['loss'] for t in trials.trials]
    x = range(len(y))
    y_err = [t['result']['ste'] * 2. for t in trials.trials]
    ax.errorbar(x, y, yerr=y_err)
    plt.savefig('obj-{agent}.png'.format(agent=args.agent_id))


