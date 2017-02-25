__author__ = 'anushabala'
from src.basic.event import Event
from src.basic.dataset import Example
from datetime import datetime
from src.lib import logstats
from src.model.vocab import is_entity
from collections import defaultdict
from itertools import izip
from src.model.preprocess import word_to_num
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import izip
import numpy as np

font_size = 12
matplotlib.rcParams.update(
    {k: font_size for k in ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')})

NO_ALPHA_MENTION = 'no_mention'
alpha_labels_to_values = {NO_ALPHA_MENTION: -1., 'least_uniform': 0., 'medium': 0.5, 'most_uniform': 1.}
alpha_values_to_labels = {-1.: NO_ALPHA_MENTION, 0.: 'least_uniform', 0.5: 'medium', 1.: 'most_uniform'}

START = '<s>'
END = '</s>'
utterance_map = {START: 0, 0: START, END: 1, 1: END}

examples = defaultdict(list)


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


def get_entity_type(entity):
    if not is_entity(entity):
        return None
    _, (_, entity_type) = entity
    return entity_type


def get_kb_strategy(kbs, dialog):
    kb_unique_vals = [get_unique_values(kbs[0]), get_unique_values(kbs[1])]
    kb_attributes = {attr.value_type for attr in kbs[0].attributes}
    attribute_agents = {}
    attribute_mention_counts = {}
    attribute_labels = {}
    attribute_order = []
    mentioned_attributes = []
    for agent, _, entities, _ in dialog:
        for token in entities:
            attr_type = token[1]
            entity = token[0]
            if attr_type in kb_attributes and attr_type not in attribute_agents.keys():
                attribute_agents[attr_type] = agent
                attribute_order.append(attr_type)
                mentioned_attributes.append((attr_type, entity, agent))
            if attr_type not in attribute_mention_counts.keys():
                attribute_mention_counts[attr_type] = 0.
            attribute_mention_counts[attr_type] += 1.

    labeled_order = []
    for attr_type in attribute_order:
        agent = attribute_agents[attr_type]
        unique_vals = kb_unique_vals[agent]
        num_unique_vals = {attr_type: len(unique_vals[attr_type]) for attr_type in unique_vals.keys()}
        sorted_unique_vals = list(sorted(list({t for t in num_unique_vals.values()})))
        pos = sorted_unique_vals.index(num_unique_vals[attr_type])
        label = 'medium'
        if len(sorted_unique_vals) > 1:
            if pos == 0:
                label = 'least_uniform'
            elif pos == len(kbs[agent].attributes) - 1:
                label = 'most_uniform'
        else:
            label = 'least_uniform'
        labeled_order.append(label)
        attribute_labels[attr_type] = label

    #if len(labeled_order) == 0:
    #    print 'Empty labeled order', dialog

    if len(attribute_mention_counts.items()) > 0:
        most_mentioned = max(attribute_mention_counts.items(), key=lambda x: x[1])
        most_mentioned_label = attribute_labels[most_mentioned[0]]
    else:
        most_mentioned_label = NO_ALPHA_MENTION

    return labeled_order, mentioned_attributes, most_mentioned_label


def abstract_entity(dialog):
    '''
    Convert entities ot integers. Different numbers corresponds to different entitys
    in _this_ dialogue.
    '''
    entity_map = {}
    new_dialog = []
    for agent, act, entities, utterance in dialog:
        m = entity_map
        for entity in entities:
            if entity not in m:
                m[entity] = len(m)
        if len(new_dialog) == 0 or new_dialog[-1][0] != agent:
            new_dialog.append((agent, act, tuple([m[e] for e in entities]), utterance))
        else:
            prev_agent, prev_act, prev_entities, prev_utterance = new_dialog[-1]
            assert agent == prev_agent
            act = tuple(sorted(list(set(act + prev_act))))
            entities = tuple(sorted(list(set(list(prev_entities) + [m[e] for e in entities]))))
            utterance = prev_utterance.append(utterance) if prev_utterance else utterance
            new_dialog[-1] = (agent, act, entities, utterance)
    return new_dialog


def print_example(name, n):
    print 'Examples for', name
    try:
        exs = examples[name]
        for ex in random.sample(exs, min(n, len(exs))):
            print ex
    except KeyError:
        print 'No example for', name


def get_utterance(dialog):
    '''
    Return a list of events/utterances in the dialogue.
    '''
    utterances = [((START,), ())]
    for agent, act, ents, utterance in dialog:
        u = (act, ents)
        utterances.append(u)
        examples[u].append(utterance)
    utterances.append(((END,), ()))
    return utterances


def get_dialog_stats(summary_map, utterance_counts, dialog):
    '''
    Count number of entities and attributes per dialogue.
    '''
    num_entities = 0
    all_ents = set()
    for agent, act, ents, utterance in dialog:
        num_ents = len(ents)
        num_types = len(set(ents))
        num_entities += num_ents
        all_ents.update(ents)
        if num_ents > 0:
            logstats.update_summary_map(summary_map, {'multi_entity_per_entity_utterance': 1 if num_types > 1 else 0})
            logstats.update_summary_map(summary_map,
                                        {'repeated_entity_per_entity_utterance': 1 if num_ents > num_types else 0})
            if num_ents > num_types:
                examples['repeated_entity_per_entity_utterance'].append(utterance)

    logstats.update_summary_map(summary_map, {'num_entity_per_dialog': num_entities,
                                              'num_entity_type_per_dialog': len(all_ents),
                                              'num_attr_type_per_dialog': len(set([e[1] for e in all_ents]))})

    dialog = abstract_entity(dialog)
    utterances = get_utterance(dialog)
    for a, b in izip(utterances, utterances[1:]):
        utterance_counts[a][b] += 1


def entity_to_type(tokens):
    return [x if not is_entity(x) else '<%s>' % x[1][1] for x in tokens]


def to_number(token, max_number):
    if token in [str(x) for x in range(max_number)]:
        return int(token)
    elif token in word_to_num:
        return word_to_num[token]
    elif token == 'all':
        return max_number
    elif token in ('none', 'no', "don't"):
        return 0
    return None


def count_kb_entity(kb, entities):
    count = 0
    for item in kb.items:
        item_entities = [x.lower() for x in item.values()]
        match = True
        for entity in entities:
            if entity not in item_entities:
                match = False
                break
        if match:
            count += 1
    return count


def check_fact(summary_map, tokens, kb):
    '''
    Simple fact checker:
        each utterance is converted to a list of numbers and entities and we assume
        that the number describes the following entities, which will cause some false
        negatives.
    '''
    hypothesis = []
    N = len(kb.items)
    for token in tokens:
        if is_entity(token):
            if len(hypothesis) > 0:
                # Represent entity as its canonical form
                hypothesis[-1][1].append(token[1][0])
        else:
            number = to_number(token, N)
            if number:
                hypothesis.append((number, []))
    for n, entities in hypothesis:
        if len(entities) > 0:
            correct = 1 if n == count_kb_entity(kb, entities) else 0
            logstats.update_summary_map(summary_map, {'correct': correct})


def update_ngram_counts(counts, utterance):
    tokens = [x if not is_entity(x) else ('<%s>' % x[1][1]) for x in utterance]
    for x in tokens:
        counts[1][(x,)] += 1
    for x, y in izip(tokens, tokens[1:]):
        counts[2][(x, y)] += 1
    for x, y, z in izip(tokens, tokens[1:], tokens[2:]):
        counts[3][(x, y, z)] += 1
    return counts

def count_to_entropy(counts):
    #print counts.keys()
    total = float(sum(counts.values()))
    probs = np.array([v / total for k, v in counts.iteritems()])
    entropy = np.dot(np.log(probs), probs) * -1.
    return entropy


def normalize(name, all_counts):
    min_ = min(all_counts.values())
    max_ = max(all_counts.values())
    n = float(all_counts[name])
    return (n - min_) / (max_ - min_ + 0.1)


def get_attr_prop(attr_name, entity_name, kb):
    '''
    Return properties of an attribute and an entity.
    - relative_domain_size: number of unique entities for the attribute
    - relative_entity_count: number of occurence of the entity
    Both are normalized between 0 and 1.
    '''
    attr_values = defaultdict(set)
    entity_counts = defaultdict(int)
    all_attrs = [attr.name for attr in kb.attributes]
    for attr in all_attrs:
        for item in kb.items:
            attr_values[attr].add(item[attr])
            entity_counts[item[attr].lower()] += 1
    attr_uniq_value_counts = {k: len(v) for k, v in attr_values.iteritems()}
    relative_domain_size = normalize(attr_name, attr_uniq_value_counts)
    relative_entity_count = normalize(entity_name, entity_counts)
    return {'relative_domain_size': relative_domain_size,
            'relative_entity_count': relative_entity_count}

def get_entity_mention(summary_map, dialog, kbs):
    '''
    Get the first mentioned entity.
    '''
    type_to_attr_name = {attr.value_type: attr.name for attr in kbs[0].attributes}
    num_mention = defaultdict(int)
    if 'first' not in summary_map:
        summary_map['first'] = defaultdict(list)
    for i, (agent, _, entities, _) in enumerate(dialog):
        for j, entity in enumerate(entities):
            attr_name = type_to_attr_name[entity[1]]
            if entity[0] not in kbs[agent].entity_set or entity[1] not in type_to_attr_name:
                continue
            if len(num_mention) == 0:
                first_mentioned_attr = (attr_name, entity[0], kbs[agent])
            num_mention[attr_name] += 1
    if len(num_mention) > 0:
        attr_props = get_attr_prop(*first_mentioned_attr)
        for k, v in attr_props.iteritems():
            summary_map['first'][k].append(v)

def analyze_strategy(all_chats, scenario_db, preprocessor, text_output, lm):
    fout = open(text_output, 'w') if text_output is not None else None
    speech_act_summary_map = defaultdict(int)
    kb_strategy_summary_map = {}
    dialog_summary_map = {}
    fact_summary_map = {}
    utterance_counts = defaultdict(lambda : defaultdict(int))
    ngram_counts = defaultdict(lambda : defaultdict(int))
    template_summary_map = {'total': 0.}
    speech_act_sequence_summary_map = {'total': 0.}
    alpha_stats = []
    num_items_stats = []
    num_attrs_mentioned = 0.
    most_mentioned_attrs = 0.
    entity_mention_summary_map = {}

    total_events = 0
    total_dialogues = 0.

    lm_summary_map = {}
    for raw in all_chats:
        ex = Example.from_dict(scenario_db, raw)
        kbs = ex.scenario.kbs
        if ex.outcome is None or ex.outcome["reward"] == 0:
            continue  # skip incomplete dialogues
        total_dialogues += 1.
        dialog = []
        mentioned_entities = set()
        for i, event in enumerate(ex.events):
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
                    logstats.update_summary_map(dialog_summary_map, {'utterance_length': len(utterance)})
                    check_fact(fact_summary_map, utterance, kbs[event.agent])
                    if lm:
                        logstats.update_summary_map(lm_summary_map,
                                                    {'score': lm.score(' '.join(entity_to_type(utterance)))})
                    update_ngram_counts(ngram_counts, utterance)
                    if fout:
                        fout.write('%s\n' % (' '.join(entity_to_type(utterance))))
            else:
                raise ValueError('Unknown event action %s.' % event.action)

            total_events += 1

            speech_act = get_speech_act(speech_act_summary_map, event, utterance)
            get_linguistic_template(template_summary_map, utterance)
            entities = [x[1] for x in utterance if is_entity(x)]
            dialog.append((event.agent, speech_act, entities, utterance))

        get_dialog_stats(dialog_summary_map, utterance_counts, dialog)
        get_speech_act_histograms(speech_act_sequence_summary_map, dialog)
        get_entity_mention(entity_mention_summary_map, dialog, kbs)

        orders, mentioned_attrs, most_mentioned_label = get_kb_strategy(kbs, dialog)
        orders = tuple(orders)
        most_mentioned_attrs += alpha_labels_to_values[most_mentioned_label]

        if len(orders) not in kb_strategy_summary_map.keys():
            kb_strategy_summary_map[len(orders)] = {}

        if orders not in kb_strategy_summary_map[len(orders)].keys():
            kb_strategy_summary_map[len(orders)][orders] = 0.0

        kb_strategy_summary_map[len(orders)][tuple(orders)] += 1.0
        alphas = ex.scenario.alphas

        num_attrs_mentioned += len(orders) / len(alphas)

        first_mentioned_label = NO_ALPHA_MENTION
        if len(orders) > 0:
            first_mentioned_label = orders[0]

        if len(mentioned_attrs) > 0:
            first_mentioned_type, first_mentioned_attr, first_agent = mentioned_attrs[0]
            update_item_stats(num_items_stats, first_mentioned_type, first_mentioned_attr, kbs[first_agent])

            if first_mentioned_label != NO_ALPHA_MENTION:
                update_alpha_stats(alpha_stats, kbs[first_agent], first_mentioned_label)
                # print "First mentioned attribute alpha:", first_mentioned, alpha_labels_to_values[first_mentioned]

    if fout:
        fout.close()
    # Summarize stats
    total = float(total_events)
    kb_strategy_totals = {k1: sum(v2 for v2 in v1.values()) for k1, v1 in kb_strategy_summary_map.items()}
    dialog_stats = {k: dialog_summary_map[k]['mean'] for k in dialog_summary_map}
    dialog_stats['entity_type_token_ratio'] = dialog_summary_map['num_entity_type_per_dialog']['sum'] / float(
        dialog_summary_map['num_entity_per_dialog']['sum'])

    unigram_counts = {k[0]: v for k, v in ngram_counts[1].iteritems()}
    dialog_stats['vocab_size'] = len(unigram_counts)
    dialog_stats['unigram_entropy'] = count_to_entropy(unigram_counts)
    multi_speech_act = sum([speech_act_summary_map[k] for k in speech_act_summary_map if len(k) > 1]) / total

    return {'speech_act': {k: speech_act_summary_map[k] / total for k in speech_act_summary_map.keys()},
            'kb_strategy': {k1: {", ".join(k2): v2 / kb_strategy_totals[k1] for k2, v2 in v1.items()} for k1, v1 in
                            kb_strategy_summary_map.items()},
            'dialog_stats': dialog_stats,
            'lm_score': -1 if not lm else lm_summary_map['score']['mean'],
            'utterance_counts': utterance_counts,
            'ngram_counts': ngram_counts,
            'linguistic_templates': template_summary_map,
            'speech_act_sequences': speech_act_sequence_summary_map,
            'correct': fact_summary_map['correct']['mean'],
            'entity_mention': {k: np.mean(v) for k, v in entity_mention_summary_map['first'].iteritems()},
            'multi_speech_act': multi_speech_act,
            'alpha_stats': alpha_stats,
            'num_items_stats': num_items_stats
            }


def update_alpha_stats(alpha_stats, kb, first_mentioned_label):
    num_items = len(kb.items)
    first_mentioned_val = alpha_labels_to_values[first_mentioned_label]

    alpha_stats.append((first_mentioned_val, num_items))


def update_item_stats(item_stats_map, first_mentioned_type, first_mentioned_attribute, kb):
    num_items = len(kb.items)
    attr_type_to_name = dict((attr.value_type, attr.name) for attr in kb.attributes)
    first_mentioned_name = attr_type_to_name[first_mentioned_type]
    repeated_vals = {}
    unique_vals = set()
    for item in kb.items:
        key = item[first_mentioned_name].lower()
        unique_vals.add(key)
        if key not in repeated_vals.keys():
            repeated_vals[key] = 0.
        repeated_vals[key] += 1.

    occurrences_set = set(repeated_vals.values())
    sorted_occurrences = list(sorted(list(occurrences_set)))
    if first_mentioned_attribute in repeated_vals.keys():
        pos = sorted_occurrences.index(repeated_vals[first_mentioned_attribute]) + 1
        item_stats_map.append((float(pos) / float(len(sorted_occurrences)), num_items))


def plot_alpha_stats(alpha_stats, save_path):
    plt.figure()
    data_with_counts = {}
    for key in alpha_stats:
        if key not in data_with_counts.keys():
            data_with_counts[key] = 0.

        data_with_counts[key] += 1.

    max_size = 2500
    x_y = [x for x in data_with_counts.keys()]
    x_vals = [x[1] for x in x_y]
    y_vals = [x[0] for x in x_y]
    total = len(alpha_stats)
    sizes = [max_size * data_with_counts[key] / total for key in x_y]
    plt.scatter(x_vals, y_vals, c='m', s=sizes, alpha=0.5)

    sorted_x = [x for x in sorted(alpha_values_to_labels.keys()) if x != -1.]
    plt.yticks(sorted_x, [alpha_values_to_labels[x] for x in sorted_x], rotation=45)
    plt.ylabel("Uniformness of first mentioned attribute")
    plt.xlabel("# of items in KB")
    plt.savefig(save_path)


def plot_num_item_stats(item_stats_map, save_path):
    plt.figure()

    data_with_counts = {}
    for (key, num_items) in item_stats_map:
        if (key, num_items) not in data_with_counts.keys():
            data_with_counts[(key, num_items)] = 0.0
        data_with_counts[(key, num_items)] += 1.
    max_size = 5000
    x_y = [x for x in data_with_counts.keys()]
    x_vals = [x[1] for x in x_y]
    y_vals = [x[0] for x in x_y]
    total = sum([data_with_counts[key] for key in x_y])
    sizes = [max_size * data_with_counts[key] / total for key in x_y]
    plt.scatter(x_vals, y_vals, c='m', s=sizes, alpha=0.5)
    plt.yticks([0.05, 0.55, 1.0], ["least_frequent", "medium", "most_frequent"], rotation=45)
    plt.ylabel("Frequency of first mentioned attribute")
    plt.xlabel("# of items in KB")
    plt.savefig(save_path)


def get_cross_talk(all_chats):
    summary_map = {}
    is_null = lambda x: x is None or x == 'null'
    count = 0

    def is_valid(event):
        if is_null(event.start_time) or event.start_time >= event.time:
            return False
        return True

    for chat in all_chats:
        if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
            events = [Event.from_dict(e) for e in chat["events"]]
            for event1, event2 in izip(events, events[1:]):
                # start_time is not available
                if not is_valid(event2):
                    continue
                sent_time = float(event1.time)
                start_time = float(event2.start_time)
                cross_talk = 1 if start_time < sent_time else 0
                logstats.update_summary_map(summary_map, {'cross_talk': cross_talk})

                if is_valid(event1):
                    typing_time = float(event1.time) - float(event1.start_time)
                    assert typing_time > 0
                    msg_len = len(event1.data)
                    logstats.update_summary_map(summary_map, {'char_per_sec': msg_len / typing_time})

    try:
        print 'Char/Sec:', summary_map['char_per_sec']['mean']
    except KeyError:
        pass
    try:
        return summary_map['cross_talk']['mean']
    # Cross talk only available for chats with start_time
    except KeyError:
        return -1


def get_linguistic_template(template_summary_map, utterance):
    if len(utterance) == 0:
        return
    template = []
    for token in utterance:
        if is_entity(token):
            template.append('<%s>' % get_entity_type(token))
        else:
            # if token not in stopwords.words('english'):
            template.append(token)

    k = tuple(template)
    if k not in template_summary_map.keys():
        template_summary_map[k] = 0.

    template_summary_map['total'] += 1.
    template_summary_map[k] += 1.


def get_speech_act_histograms(speech_act_sequence_summary_map, dialog, collapsed=False):
    seq = []
    last_act = None
    for (_, act, _, _) in dialog:
        if (act != last_act and collapsed) or (not collapsed):
            seq.append(act)
        last_act = act

    k = tuple(seq)
    if k not in speech_act_sequence_summary_map.keys():
        speech_act_sequence_summary_map[k] = 0.

    speech_act_sequence_summary_map['total'] += 1.
    speech_act_sequence_summary_map[k] += 1.


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
                    total_time_taken += (end_time - start_time)
                except ValueError:
                    print "Error parsing event times: %s, %s" % (events[0].time, events[-1].time)

                total_complete += 1
    if total_complete == 0:
        # no complete dialogues for this setting - should never happen with sufficient data
        print "No complete dialogues for ", alphas
        return -1.0
    return total_time_taken / total_complete


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
    return total_length / total_sentences


def get_average_sentences(all_chats, scenario_db, alphas=None, num_items=None):
    total_length = 0.0
    total_complete = 0.0
    max_length = 0
    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            l = len(chat['events'])
            max_length = max(l, max_length)
            if chat["outcome"] is not None and chat["outcome"]["reward"] == 1:
                total_length += l
                total_complete += 1
    if total_complete == 0:
        # no complete dialogues for this setting - should never happen with sufficient data
        print "No complete dialogues for ", alphas
        return -1.0
    return total_length / total_complete


def get_turns_vs_completed(all_chats):
    num_turns_dict = defaultdict(dict)
    for chat in all_chats:
        if chat["outcome"] is not None:
            num_turns = len(chat['events'])
            logstats.update_summary_map(num_turns_dict[num_turns],
                                        {'complete': 1 if chat["outcome"]["reward"] == 1 else 0})
    return {k: v['complete']['sum'] for k, v in num_turns_dict.iteritems()}


def get_select_vs_completed(all_chats):
    num_select_dict = defaultdict(dict)
    for chat in all_chats:
        if chat["outcome"] is not None:
            events = [Event.from_dict(e) for e in chat["events"]]
            num_select = len([e for e in events if e.action == 'select'])
            logstats.update_summary_map(num_select_dict[num_select],
                                        {'complete': 1 if chat["outcome"]["reward"] == 1 else 0})
    return {k: v['complete']['sum'] for k, v in num_select_dict.iteritems()}


def get_average_select(all_chats):
    num_select = 0
    num_chat = 0
    for chat in all_chats:
        if chat["outcome"] is not None:
            events = [Event.from_dict(e) for e in chat["events"]]
            num_select += len([e for e in events if e.action == 'select'])
            num_chat += 1
    return num_select / float(num_chat)


def get_num_completed(all_chats, scenario_db, alphas=None, num_items=None):
    num_complete = 0.0
    for chat in all_chats:
        scenario = scenario_db.get(chat["scenario_uuid"])
        kb = scenario.get_kb(0)
        items = len(kb.items)
        if (alphas is not None and tuple(scenario.alphas) == alphas) \
                or (num_items is not None and items == num_items) \
                or (alphas is None and num_items is None):
            if chat["outcome"] is not None:
                num_complete += 1.0 if chat["outcome"]["reward"] == 1 else 0.0

    return num_complete


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
    stats = {
        'avg_time_taken': get_average_time_taken(all_chats, scenario_db),
        'avg_turns': get_average_sentences(all_chats, scenario_db),
        'avg_select': get_average_select(all_chats),
        'turns_vs_completed': get_turns_vs_completed(all_chats),
        'select_vs_completed': get_select_vs_completed(all_chats),
        'avg_sentence_length': get_average_length(all_chats, scenario_db),
        'num_completed': get_num_completed(all_chats, scenario_db),
        'cross_talk': get_cross_talk(all_chats),
        'total': get_total(all_chats, scenario_db)
    }
    total = float(stats['total'])
    for t in stats['turns_vs_completed']:
        stats['turns_vs_completed'][t] /= total
    for t in stats['select_vs_completed']:
        stats['select_vs_completed'][t] /= total
    stats['completion_rate'] = stats['num_completed'] / total
    stats['completion_per_turn'] = stats['completion_rate'] / stats['avg_turns']
    stats['completion_per_select'] = stats['completion_rate'] / stats['avg_select']
    return stats


def print_group_stats(group_stats):
    print "Average time taken per dialogue: %2.2f seconds" % group_stats['avg_time_taken']
    print "Average number of utterances: %2.2f" % group_stats['avg_turns']
    print "Average utterance length: %2.2f tokens" % group_stats['avg_sentence_length']
    print "# of completed dialogues: %d" % group_stats['num_completed']
    try:
        print "%% of cross talk: %.2f" % group_stats['cross_talk']
    # cross_talk is not computed for alpha groups for now
    except KeyError:
        pass
    print 'Total dialogues: %d' % group_stats['total']


def get_topk_utterance(n, items):
    total = float(sum([x[1] for x in items]))
    sorted_counts = sorted(items, key=lambda x: x[1], reverse=True)
    result = []
    for k, v in sorted_counts[:n]:
        item = (k, v / total)
        result.append(item)
    return result, len(sorted_counts), sum([x[1] for x in result])


def get_initial_utterance(n, counts):
    start = ((START,), ())
    init_counts = counts[start]
    return get_topk_utterance(n, init_counts.items())


def get_unigram_utterance(n, counts):
    start = ((START,), ())
    unigram_counts = [(k, sum(v.values())) for k, v in counts.iteritems() if k != start]
    return get_topk_utterance(n, unigram_counts)


def get_bigram_utterance(n, counts):
    bigram_counts = [((k1, k2), v) for k1, d in counts.iteritems() for k2, v in d.iteritems()]
    return get_topk_utterance(n, bigram_counts)


def get_top_k_from_counts(n, counts):
    """
    Given a map of counts mapping from a key to its frequency, returns the top k keys (based on frequency) after
    normalizing the frequencies by the total.
    :param n: The number of keys to return
    :param counts: A map of counts mapping from a key to its frequency.
    :return: A map from every key to its normalized frequency
    """
    total = sum(counts.values())
    sorted_counts = sorted([(k, v / total) for (k, v) in counts.items() if k != 'total'], key=lambda x: x[1],
                           reverse=True)
    return sorted_counts[:n]


def print_strategy_stats(stats):
    speech_act_stats = stats['speech_act']
    dialogue_stats = stats['dialog_stats']
    utterance_counts = stats['utterance_counts']
    ngram_counts = stats['ngram_counts']
    template_counts = stats['linguistic_templates']
    speech_act_sequences = stats['speech_act_sequences']

    print "-----------------------------------"
    print 'Vocabulary size:', dialogue_stats['vocab_size']
    print 'Unigram entropy:', dialogue_stats['unigram_entropy']

    print "-----------------------------------"
    print 'Speech act statistics:'
    for act_type, frac in sorted([(a, b) for a, b in speech_act_stats.items()], key=lambda x: x[1], reverse=True):
        print '%% %s: %2.3f' % (act_type, frac)
    print 'multi speech acts:', stats['multi_speech_act']

    print "-----------------------------------"
    print 'Dialogue statistics:'
    for k, v in dialogue_stats.iteritems():
        print '%s: %.3f' % (k, v)
    print_example('repeated_entity_per_entity_utterance', 10)

    k = 10
    print "-----------------------------------"
    print 'Top %d ngrams:' % (k,)
    for n in xrange(1, 4):
        sorted_words = sorted(ngram_counts[n].iteritems(), key=lambda x: x[1], reverse=True)
        total = float(sum(ngram_counts[n].values()))
        print '----- n = %d -----' % n
        for i in xrange(min(k, len(sorted_words))):
            word, count = sorted_words[i]
            print '%s: %.3f' % (word, count / total)

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
    print "Number of linguistic templates: %d" % len(template_counts.keys())
    k = 10
    print 'Top %d linguistic templates' % k
    top_templates = get_top_k_from_counts(k, template_counts)
    for template, v in top_templates:
        print '%s: %.3f' % (" ".join(template), v)
    print '# templates:', len(template_counts)

    k = 10
    print "-----------------------------------"
    print 'Top %d speech act sequences' % k
    top_speech_act_sequences = get_top_k_from_counts(k, speech_act_sequences)
    for template, v in top_speech_act_sequences:
        print '[%s]: %.3f' % (" ".join([str(t) for t in template]), v)
