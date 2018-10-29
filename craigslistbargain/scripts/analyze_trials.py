import argparse
from collections import defaultdict
import numpy as np
from scipy.stats import sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cocoa.core.util import read_pickle
from cocoa.core.dataset import read_examples
from core.scenario import Scenario
from sessions.rulebased_session import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-db', help='.pkl file of config evaluation results')
    parser.add_argument('--transcripts', nargs='*')
    args = parser.parse_args()

    examples = read_examples(args.transcripts, -1, Scenario)
    examples = {e.ex_id: e for e in examples}

    trials = read_pickle(args.trial_db)
    x = []
    margins = defaultdict(list)
    init_price = defaultdict(list)
    success = defaultdict(list)
    humanlike = defaultdict(list)
    scenarios = set()
    data = []
    for config, results in sorted(trials.items(), key=lambda x: x[0]):
        config = Config(*config)
        x.append(config.overshoot)
        num_success = defaultdict(int)
        num_results = defaultdict(int)
        config_margins = defaultdict(list)
        config_init_price = defaultdict(list)
        config_humanlike = defaultdict(list)
        for chat_id, obj_values in results.iteritems():
            if obj_values['margin'] is not None and obj_values['margin'] > 10:
                continue
            example = examples[chat_id]
            kbs = example.scenario.kbs
            target = kbs[0].target
            listing_price = kbs[0].listing_price
            init_price = listing_price / ((1. - config.overshoot) * target)
            scenario_id = obj_values['scenario_id']
            scenarios.add(scenario_id)
            config_init_price[scenario_id].append(init_price)
            num_results[scenario_id] += 1
            if obj_values['margin'] is not None and not np.isnan(obj_values['margin']):
                num_success[scenario_id] += 1
                config_margins[scenario_id].append(obj_values['margin'])
                margin = obj_values['margin']
                final_price = float(listing_price) / example.outcome['offer']['price']
            else:
                margin = None
                final_price = None
            config_humanlike[scenario_id].append(float(obj_values['humanlike']))
            row = {
                    'overshoot': config.overshoot,
                    'scenario_id': scenario_id,
                    'init_price': init_price,
                    'margin': margin,
                    'final_price': final_price,
                    'success': 1 if margin is not None else 0,
                    'humanlike': float(obj_values['humanlike']),
                    }
            data.append(row)
        #print 'margins:', config_margins
        #print 'humanlike:', config_humanlike
        print 'overshoot:', config.overshoot
        print 'success:', dict(num_success), len(results)
        for scenario_id in scenarios:
            #if len(config_margins[scenario_id]) == 0:
            #    margins[scenario_id].append((0, 0))
            #else:
            #    margins[scenario_id].append((np.mean(config_margins[scenario_id]), sem(config_margins[scenario_id])))
            #humanlike[scenario_id].append((np.mean(config_humanlike[scenario_id]), sem(config_humanlike[scenario_id])))
            margins[scenario_id].append(config_margins[scenario_id])
            humanlike[scenario_id].append(config_humanlike[scenario_id])
            success[scenario_id].append(float(num_success[scenario_id]) / num_results[scenario_id])

    plot_data = {}
    for row in data:
        if row['scenario_id'] not in plot_data:
            plot_data[row['scenario_id']] = defaultdict(lambda : defaultdict(list))
        plot_data[row['scenario_id']][row['init_price']]['final_price'].append(row['final_price'])
        plot_data[row['scenario_id']][row['init_price']]['humanlike'].append(row['humanlike'])
        plot_data[row['scenario_id']][row['init_price']]['success'].append(row['success'])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    init_prices = []
    for i, scenario_id in enumerate(scenarios):
        init_prices.extend(plot_data[scenario_id].keys())
        init_prices = sorted(init_prices)
    final_prices = defaultdict(list)
    humanlike = defaultdict(list)
    for init_price in init_prices:
        for i, scenario_id in enumerate(scenarios):
            for x in plot_data[scenario_id][init_price]['final_price']:
                if x is not None:
                    #final_prices.append((init_price, x))
                    final_prices[init_price].append(x)
            for x in plot_data[scenario_id][init_price]['humanlike']:
                if x is not None:
                    humanlike[init_price].append(x)
    #axes.scatter([x[0] for x in final_prices], [x[1] for x in final_prices])
    final_prices = [final_prices[init_price] for init_price in init_prices]
    humanlike = [humanlike[init_price] for init_price in init_prices]
    print humanlike
    axes[0].boxplot(final_prices)
    axes[0].set_ylabel('final_price')
    axes[1].boxplot(humanlike)
    axes[1].set_ylabel('final_price')
    plt.savefig('trials3.png')
    import sys; sys.exit()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 5))
    for i, scenario_id in enumerate(scenarios):
        #axes[0].errorbar(x, [m[0] for m in margins[scenario_id]], yerr=[m[1] for m in margins[scenario_id]], label=scenario_id, fmt='o')
        axes[i][0].boxplot(margins[scenario_id])
        axes[i][0].set_ylabel('margins')

        #axes[1].errorbar(x, [h[0] for h in humanlike[scenario_id]], yerr=[h[1] for h in humanlike[scenario_id]], label=scenario_id, fmt='o')
        axes[i][1].boxplot(humanlike[scenario_id])
        axes[i][1].set_ylabel('humanlike')

        axes[i][2].plot(x, success[scenario_id], label=scenario_id)
        axes[i][2].set_ylabel('success_rate')

    plt.legend()
    plt.savefig('trials3.png')


