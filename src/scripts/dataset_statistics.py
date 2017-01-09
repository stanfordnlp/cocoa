__author__ = 'anushabala'
from src.basic.event import Event
from src.basic.dataset import Example
from datetime import datetime
from src.lib import logstats
from src.model.vocab import is_entity
from collections import defaultdict
from itertools import izip
import random
import matplotlib.pyplot as plt


def is_question(tokens):
    first_word = tokens[0]
    last_word = tokens[-1]
    if last_word == '?' or first_word in ('do', 'does', 'what', 'any'):
        return True
    return False

def is_inform(tokens):
    for token in tokens:
        if is_entity(token):
            return True
    return False

def match_keywords(tokens):
    types = []
    for token in tokens:
        if token in ('yes', 'yep', 'yeah', 'no', 'nope', 'none'):
            types.append('answer')
        elif token in ('hi', 'hello', 'hey', 'hiya', 'howdy'):
            types.append('greeting')
        elif token in ('sorry',):
            types.append('apology')
    return types

def get_speech_act(summary_map, event, utterance):
    act = []
    if event.action == 'select':
        act.append('select')
    elif event.action == 'message':
        if is_question(utterance):
            act.append('question')
        else:
            # NOTE: inform and answer are not exclusive, e.g. no, I have 3 apple.
            inform = is_inform(utterance)
            other_acts = match_keywords(utterance)
            if inform:
                act.append('inform')
            act.extend(other_acts)
    if len(act) == 0:
        act = ('other',)
    else:
        act = tuple(sorted(list(set(act))))
    summary_map[act] += 1
    return act

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
    for agent, _, entities, _ in dialog:
        for token in entities:
            attr_type = token[1]
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

def abstract_entity(dialog):
    #entity_map = {0: {}, 1: {}}
    #entity_map = {}
    new_dialog = []
    for agent, act, entities, utterance in dialog:
        #m = entity_map[agent]
        #m = entity_map
        m = {}
        for entity in entities:
            if entity not in m:
                m[entity] = len(m)
        new_dialog.append((agent, act, tuple([m[e] for e in entities]), utterance))
    return new_dialog

START = '<s>'
END = '</s>'
utterance_map = {START: 0, 0: START, END: 1, 1: END}

examples = defaultdict(list)

def print_example(name, n):
    print 'Examples for', name
    try:
        exs = examples[name]
        for ex in random.sample(exs, min(n, len(exs))):
            print ex
    except KeyError:
        print 'No example for', name

def map_utterance(dialog):
    '''
    Convert a list of events/utterances to integers.
    '''
    utterances = [utterance_map[START]]
    for agent, act, ents, utterance in dialog:
        u = (act, ents)
        if u not in utterance_map:
            id_ = len(utterance_map)
            utterance_map[u] = id_
            utterance_map[id_] = u
        utterances.append(utterance_map[u])
        examples[u].append(utterance)
    utterances.append(utterance_map[END])
    return utterances

def get_dialog_stats(summary_map, utterance_counts, dialog):
    num_entities = 0
    num_entity_types = 0
    for agent, act, ents, utterance in dialog:
        num_ents = len(ents)
        num_types = len(set(ents))
        num_entities += num_ents
        num_entity_types += num_types
        if num_ents > 0:
            logstats.update_summary_map(summary_map, {'multi_entity_per_entity_utterance': 1 if num_ents > 1 else 0})
            logstats.update_summary_map(summary_map, {'repeated_entity_per_entity_utterance': 1 if num_ents > num_types else 0})
            if num_ents > num_types:
                examples['repeated_entity_per_entity_utterance'].append(utterance)

    logstats.update_summary_map(summary_map, {'num_entity_per_dialog': num_entities, 'num_entity_type_per_dialog': num_entity_types})

    dialog = abstract_entity(dialog)
    int_utterances = map_utterance(dialog)
    for a, b in izip(int_utterances, int_utterances[1:]):
        utterance_counts[a][b] += 1

def analyze_strategy(all_chats, scenario_db, preprocessor):
    speech_act_summary_map = defaultdict(int)
    kb_strategy_summary_map = {}
    dialog_summary_map = {}
    utterance_counts = defaultdict(lambda : defaultdict(int))
    first_word_counts = defaultdict(int)
    total_events = 0
    for raw in all_chats:
        ex = Example.from_dict(scenario_db, raw)
        kbs = ex.scenario.kbs
        if ex.outcome is None or ex.outcome["reward"] == 0:
            continue  # skip incomplete dialogues
        dialog = []
        for i, event in enumerate(ex.events):
            if event.action == 'select':
                utterance = []
                if i == 0:
                    first_word_counts['<select>'] += 1
            elif event.action == 'message':
                utterance = preprocessor.process_event(event, kbs[event.agent])
                # Skip empty utterances
                if not utterance:
                    continue
                else:
                    utterance = utterance[0]
                    if i == 0:
                        first_word_counts[utterance[0]] += 1
            else:
                raise ValueError('Unknown event action %s.' % event.action)

            total_events += 1

            speech_act = get_speech_act(speech_act_summary_map, event, utterance)
            entities = [x[1] for x in utterance if is_entity(x)]
            dialog.append((event.agent, speech_act, entities, utterance))

        get_dialog_stats(dialog_summary_map, utterance_counts, dialog)

        orders = tuple(get_kb_strategy(kbs, dialog))
        if len(orders) not in kb_strategy_summary_map.keys():
            kb_strategy_summary_map[len(orders)] = {}

        if orders not in kb_strategy_summary_map[len(orders)].keys():
            kb_strategy_summary_map[len(orders)][orders] = 0.0

        kb_strategy_summary_map[len(orders)][tuple(orders)] += 1.0

    # Summarize stats
    total = float(total_events)
    kb_strategy_totals = {k1: sum(v2 for v2 in v1.values()) for k1, v1 in kb_strategy_summary_map.items()}
    return {'speech_act': {k: speech_act_summary_map[k] / total for k in speech_act_summary_map.keys()},
            'kb_strategy': {k1: {", ".join(k2): v2/kb_strategy_totals[k1] for k2, v2 in v1.items()} for k1, v1 in kb_strategy_summary_map.items()},
            'dialog_stats': {k: dialog_summary_map[k]['mean'] for k in dialog_summary_map},
            'utterance_counts': utterance_counts,
            'first_word_counts': first_word_counts,
            }

def get_cross_talk(all_chats):
    summary_map = {}
    is_null = lambda x: x is None or x == 'null'
    for chat in all_chats:
        if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
            events = [Event.from_dict(e) for e in chat["events"]]
            # start_time is not recorded
            if events[0].action != 'select' and is_null(events[0].start_time):
                continue
            for event1, event2 in izip(events, events[1:]):
                sent_time = float(event1.time)
                # start_time is None for select
                start_time = float(event2.start_time) if not is_null(event2.start_time) else float(event2.time)
                cross_talk = 1 if start_time < sent_time else 0
                logstats.update_summary_map(summary_map, {'cross_talk': cross_talk})
    return summary_map['cross_talk']['mean']


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
                try:
                    start_time = float(events[0].time)
                    end_time = float(events[-1].time)
                    total_time_taken += (end_time-start_time)
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
        'cross_talk': get_cross_talk(all_chats),
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
    print "%% of cross talk: %.2f" % group_stats['cross_talk']
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

def get_topk_utterance(n, items):
    total = float(sum([x[1] for x in items]))
    sorted_counts = sorted(items, key=lambda x: x[1], reverse=True)
    result = []
    for k, v in sorted_counts[:n]:
        if isinstance(k, tuple):
            item = (tuple([utterance_map[x] for x in k]), v / total)
        else:
            item = (utterance_map[k], v / total)
        result.append(item)
    return result, len(sorted_counts), sum([x[1] for x in result])

def get_initial_utterance(n, counts):
    start = utterance_map[START]
    init_counts = counts[start]
    return get_topk_utterance(n, init_counts.items())

def get_unigram_utterance(n, counts):
    start = utterance_map[START]
    unigram_counts = [(k, sum(v.values())) for k, v in counts.iteritems() if k != start]
    return get_topk_utterance(n, unigram_counts)

def get_bigram_utterance(n, counts):
    bigram_counts = [((k1, k2), v) for k1, d in counts.iteritems() for k2, v in d.iteritems()]
    return get_topk_utterance(n, bigram_counts)

def print_strategy_stats(stats):
    speech_act_stats = stats['speech_act']
    dialogue_stats = stats['dialog_stats']
    kb_strategy_stats = stats['kb_strategy']
    utterance_counts = stats['utterance_counts']
    first_word_counts = stats['first_word_counts']

    print "-----------------------------------"
    print 'Speech act statistics:'
    for act_type, frac in sorted([(a, b) for a,b in speech_act_stats.items()], key=lambda x:x[1], reverse=True):
        print '%% %s: %2.3f' % (act_type, frac)

    print "-----------------------------------"
    print 'Dialogue statistics:'
    for k, v in dialogue_stats.iteritems():
        print '%s: %.3f' % (k, v)
    print_example('repeated_entity_per_entity_utterance', 3)

    k = 10
    print "-----------------------------------"
    print 'Top %d first words:' % (k,)
    sorted_words = sorted(first_word_counts.iteritems(), key=lambda x: x[1], reverse=True)
    total = float(sum([x[1] for x in sorted_words]))
    for i in xrange(min(k, len(sorted_words))):
        word, count = sorted_words[i]
        print '%s: %.3f' % (word, count / total)

    k = 5
    utterances, total, frac = get_initial_utterance(k, utterance_counts)
    print "-----------------------------------"
    print 'Top %d/%d/%.2f initial utterances:' % (k, total, frac)
    for u, frac in utterances:
        print '%s: %.3f' % (u, frac)

    k = 10
    utterances, total, frac = get_unigram_utterance(k, utterance_counts)
    print "-----------------------------------"
    print 'Top %d/%d/%.2f unigram utterances:' % (k, total, frac)
    for u, frac in utterances:
        print '%s: %.3f' % (u, frac)
        print_example(u, 2)

    utterances, total, frac = get_bigram_utterance(k, utterance_counts)
    print "-----------------------------------"
    print 'Top %d/%d/%.2f bigram utterances:' % (k, total, frac)
    for u, frac in utterances:
        print '%s, %s: %.3f' % (u[0], u[1], frac)

    print "-----------------------------------"
    print "KB attribute-based strategy statistics:"
    for num_attrs, v in kb_strategy_stats.items():
        print "Number of attributes mentioned: %d" % num_attrs
        for order, frac in sorted([(a, b) for a, b in v.items()], key=lambda x: x[1], reverse=True):
            print "\t%s: %2.3f" % (order, frac)


