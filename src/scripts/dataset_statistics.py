from collections import defaultdict

__author__ = 'anushabala'
from src.basic.event import Event
from src.basic.dataset import Example
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.lib import logstats
from src.model.vocab import is_entity


date_fmt = '%Y-%m-%d %H-%M-%S'

def is_question(tokens):
    first_word = tokens[0]
    last_word = tokens[-1]
    if last_word == '?' or first_word in ('do', 'does', 'what'):
        return True
    return False

def is_inform(tokens):
    for token in tokens:
        if is_entity(token):
            return True
    return False

def is_answer(tokens):
    for token in tokens:
        if token in ('yes', 'yep', 'yeah', 'no', 'nope', 'none'):
            return True
    return False

def get_speech_acts(summary_map, event, utterance):
    if event.action == 'select':
        logstats.update_summary_map(summary_map, {'select': 1})
    elif event.action == 'message':
        if is_question(utterance):
            logstats.update_summary_map(summary_map, {'question': 1})
        else:
            # NOTE: inform and answer are not exclusive, e.g. no, I have 3 apple.
            inform = is_inform(utterance)
            answer = is_answer(utterance)
            if inform:
                logstats.update_summary_map(summary_map, {'inform': 1})
            if answer:
                logstats.update_summary_map(summary_map, {'answer': 1})
            if not inform and not answer:
                logstats.update_summary_map(summary_map, {'other': 1})


def get_unique_values(kb):
    unique_vals = {}
    for attr in kb.attributes:
        attr_vals = set()
        for item in kb.items:
            attr_vals.add(item[attr.name])
        unique_vals[attr.value_type] = attr_vals
    return unique_vals


def get_kb_strategy(kbs, dialog):
    kb_unique_vals = [get_unique_values(kbs[0]), get_unique_values(kbs[1])]
    kb_attributes = {attr.value_type for attr in kbs[0].attributes}
    attribute_agents = {}
    attribute_order = []
    for agent, utterance in dialog:
        for token in utterance:
            if is_entity(token):
                attr_type = token[1][1]
                if attr_type in kb_attributes and attr_type not in attribute_agents.keys():
                    attribute_agents[attr_type] = agent
                    attribute_order.append(attr_type)

    labeled_order = []
    for attr_type in attribute_order:
        agent = attribute_agents[attr_type]
        unique_vals = kb_unique_vals[agent]
        num_unique_vals = {attr_type: len(unique_vals[attr_type]) for attr_type in unique_vals.keys()}
        sorted_unique_vals = list(sorted(list({t for t in num_unique_vals.values()})))
        pos = sorted_unique_vals.index(num_unique_vals[attr_type])
        label = 'medium'
        if pos == 0:
            label = 'least_uniform'
        elif pos == len(kbs[agent].attributes) - 1:
            label = 'most_uniform'
        labeled_order.append(label)

    return labeled_order


def analyze_strategy(all_chats, scenario_db, preprocessor):
    speech_act_summary_map = {}
    kb_strategy_summary_map = {}
    total_events = 0
    for raw in all_chats:
        ex = Example.from_dict(scenario_db, raw)
        kbs = ex.scenario.kbs
        if ex.outcome is None or ex.outcome["reward"] == 0:
            continue  # skip incomplete dialogues
        dialog = []
        for event in ex.events:
            if event.action == 'select':
                utterance = None
            elif event.action == 'message':
                utterance = preprocessor.process_event(event, kbs[event.agent])
                # Skip empty utterances
                if not utterance:
                    continue
                else:
                    utterance = utterance[0]
                    dialog.append((event.agent, utterance))
            else:
                raise ValueError('Unknown event action %s.' % event.action)

            total_events += 1
            # All analysis
            get_speech_acts(speech_act_summary_map, event, utterance)
        orders = tuple(get_kb_strategy(kbs, dialog))
        if len(orders) not in kb_strategy_summary_map.keys():
            kb_strategy_summary_map[len(orders)] = {}

        if orders not in kb_strategy_summary_map[len(orders)].keys():
            kb_strategy_summary_map[len(orders)][orders] = 0.0

        kb_strategy_summary_map[len(orders)][tuple(orders)] += 1.0


    # Summarize stats
    total = float(total_events)
    kb_strategy_totals = {k1: sum(v2 for v2 in v1.values()) for k1, v1 in kb_strategy_summary_map.items()}
    return {'speech_act': {k: speech_act_summary_map[k]['sum'] / total for k in ('question', 'inform', 'answer', 'other', 'select') if k in speech_act_summary_map.keys()},
            'kb_strategy': {k1: {", ".join(k2): v2/kb_strategy_totals[k1] for k2, v2 in v1.items()} for k1, v1 in kb_strategy_summary_map.items()}}


def get_average_time_taken(all_chats, scenario_db, alphas=None, num_items=None):
    total_time_taken = 0.0
    total_complete = 0.0

    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
                events = [Event.from_dict(e) for e in chat["events"]]

                if type(events[0].time) is int:
                    total_time_taken = events[-1].time - events[0].time
                else:
                    start_time = datetime.strptime(events[0].time, date_fmt)
                    end_time = datetime.strptime(events[-1].time, date_fmt)
                    total_time_taken += (end_time-start_time).seconds

                if isinstance(events[0].time, str):
                    try:
                        start_time = datetime.strptime(events[0].time, date_fmt)
                        end_time = datetime.strptime(events[-1].time, date_fmt)
                        total_time_taken += (end_time-start_time).seconds
                    except ValueError:
                        print "Error parsing event times: %s, %s" % (events[0].time, events[-1].time)

                total_complete += 1
    if total_complete == 0:
        # no complete dialogues for this setting - should never happen with sufficient data
        print "No complete dialogues for ", alphas
        return -1.0
    return total_time_taken/total_complete


def get_average_length(all_chats, scenario_db, alphas=None, num_items=None):
    total_length = 0.0
    total_sentences = 0.0

    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
                events = [Event.from_dict(e) for e in chat["events"]]
                for e in events:
                    if e.action == "message":
                        total_sentences += 1
                        total_length += len(e.data.split())
    if total_sentences == 0:
        # no complete dialogues for this setting - should never happen with sufficient data
        print "No complete dialogues for (alphas=", alphas, ", items=", num_items, ")"
        return -1.0
    return total_length/total_sentences


def get_average_sentences(all_chats, scenario_db, alphas=None, num_items=None):
    total_length = 0.0
    total_complete = 0.0
    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
                events = [Event.from_dict(e) for e in chat["events"]]
                total_length += len([e for e in events if e.action == 'message'])
                total_complete += 1
    if total_complete == 0:
        # no complete dialogues for this setting - should never happen with sufficient data
        print "No complete dialogues for ", alphas
        return -1.0
    return total_length/total_complete


def get_num_completed(all_chats, scenario_db, alphas=None, num_items=None):
    num_complete = 0.0
    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            num_complete += 1.0 if chat["outcome"] is not None and chat["outcome"]["reward"] == 1 else 0.0

    return num_complete


def get_alpha_groups(all_chats, scenario_db):
    scenario_groups = {}
    for chat in all_chats:
        scenario_id = chat["scenario_uuid"]
        scenario = scenario_db.get(scenario_id)
        alphas = tuple(scenario.alphas)
        if alphas not in scenario_groups.keys():
            scenario_groups[alphas] = 0.0
        scenario_groups[alphas] += 1.0

    return scenario_groups


def get_item_groups(all_chats, scenario_db):
    item_groups = {}
    for chat in all_chats:
        scenario_id = chat["scenario_uuid"]
        scenario = scenario_db.get(scenario_id)
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if items not in item_groups.keys():
            item_groups[items] = 0.0
        item_groups[items] += 1.0

    return item_groups


def get_total(all_chats, scenario_db, alphas=None, num_items=None):
    if alphas is None and num_items is None:
        return len(all_chats)

    total = 0.0
    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items):
            total += 1.0

    return total


def get_total_statistics(all_chats, scenario_db):
    return {
        'avg_time_taken': get_average_time_taken(all_chats, scenario_db),
        'avg_turns': get_average_sentences(all_chats, scenario_db),
        'avg_sentence_length': get_average_length(all_chats, scenario_db),
        'num_completed': get_num_completed(all_chats, scenario_db),
        'total': get_total(all_chats, scenario_db)
    }


def get_statistics_by_alpha(all_chats, scenario_db):
    scenario_groups = get_alpha_groups(all_chats, scenario_db)
    stats = {}

    print "Number of alpha settings: %d" % len(scenario_groups.keys())
    for alphas in scenario_groups.keys():
        group_stats = {
            'avg_time_taken': get_average_time_taken(all_chats, scenario_db, alphas=alphas),
            'avg_turns': get_average_sentences(all_chats, scenario_db, alphas=alphas),
            'avg_sentence_length': get_average_length(all_chats, scenario_db, alphas=alphas),
            'num_completed': get_num_completed(all_chats, scenario_db, alphas=alphas),
            'total': get_total(all_chats, scenario_db, alphas=alphas)
        }
        str_key = ", ".join([str(a) for a in alphas])
        stats[str_key] = group_stats

    return stats


def get_statistics_by_items(all_chats, scenario_db):
    stats = {}
    item_groups = get_item_groups(all_chats, scenario_db)
    print "Number of variations in # items: %d" % len(item_groups.keys())

    for items in item_groups.keys():
        group_stats = {
            'avg_time_taken': get_average_time_taken(all_chats, scenario_db, num_items=items),
            'avg_turns': get_average_sentences(all_chats, scenario_db, num_items=items),
            'avg_sentence_length': get_average_length(all_chats, scenario_db, num_items=items),
            'num_completed': get_num_completed(all_chats, scenario_db, num_items=items),
            'total': get_total(all_chats, scenario_db, num_items=items)
        }

        stats[items] = group_stats

    return stats


def print_group_stats(group_stats):
    print "Average time taken per dialogue: %2.2f seconds" % group_stats['avg_time_taken']
    print "Average number of utterances: %2.2f" % group_stats['avg_turns']
    print "Average utterance length: %2.2f tokens" % group_stats['avg_sentence_length']
    print "# of completed dialogues: %d" % group_stats['num_completed']
    print 'Total dialogues: %d' % group_stats['total']


def print_stats(stats, stats_type="alphas"):
    for group in sorted(stats.keys()):
        print "-----------------------------------"
        print "Statistics for %s: %s" % (stats_type, group)
        print_group_stats(stats[group])


def plot_num_items_stats(stats, save_path):
    x_values = sorted(stats.keys())
    avg_times = [stats[x]['avg_time_taken'] for x in x_values]
    avg_tokens = [stats[x]['avg_sentence_length'] for x in x_values]
    avg_utterances = [stats[x]['avg_turns'] for x in x_values]
    completed_ratio = [stats[x]['num_completed']/stats[x]['total'] for x in x_values]

    plt.plot(x_values, avg_times, 'r--', label='Average time to complete dialogue')
    plt.plot(x_values, avg_utterances, 'b-x', label='Average # of utterances')
    plt.plot(x_values, avg_tokens, 'g-.', label='Average tokens/utterance')
    plt.plot(x_values, completed_ratio, 'm-', label='Fraction of complete dialogues')
    plt.xlabel('Number of items in scenario')

    plt.legend(loc='best')
    plt.savefig(save_path)


def print_strategy_stats(stats):
    speech_act_stats = stats['speech_act']
    kb_strategy_stats = stats['kb_strategy']

    print 'Speech act statistics:'
    for act_type, frac in sorted([(a, b) for a,b in speech_act_stats.items()], key=lambda x:x[1], reverse=True):
        print '%% %s: %2.3f' % (act_type, frac)

    print "-----------------------------------"
    print "KB attribute-based strategy statistics:"
    for num_attrs, v in kb_strategy_stats.items():
        print "Number of attributes mentioned: %d" % num_attrs
        for order, frac in sorted([(a, b) for a, b in v.items()], key=lambda x: x[1], reverse=True):
            print "\t%s: %2.3f" % (order, frac)


