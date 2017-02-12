import json
import matplotlib.pyplot as plt
import numpy as np

# Name of eval results file (output of make_eval_results.py)
part1 = None
with open(part1) as f:
    part1_results = json.load(f)


num_human_evals = 0
num_rule_evals = 0
num_static_evals = 0
num_dynamic_evals = 0

# To store histogram of responses
human_responses = np.zeros(5, dtype=np.float32)
rule_responses = np.zeros(5, dtype=np.float32)
static_responses = np.zeros(5, dtype=np.float32)
dynamic_responses = np.zeros(5, dtype=np.float32)


def get_question_type_percentages(dialogue_to_responses, dialogue_to_agent_mapping, type):
    """
    For given question type (fluent, strategic, etc), get histogram for each model type.
    Return as dict from model type to np array with histogram values
    :param type:
    :return:
    """
    global num_human_evals
    global num_static_evals
    global num_dynamic_evals
    global num_rule_evals

    global human_responses, rule_responses, static_responses, dynamic_responses

    for dialogue_id, values in dialogue_to_responses.iteritems():
        agent_mapping = dialogue_to_agent_mapping[dialogue_id]
        agent_mapping = json.loads(agent_mapping)

        for agent_id, question_responses in values.iteritems():
            agent_type = agent_mapping[str(agent_id)]
            for question, responses in question_responses.iteritems():
                if question == type and agent_type == "human":
                    for r in responses:
                        human_responses[r-1] += 1
                        num_human_evals += 1
                elif question == type and agent_type == "rulebased":
                    for r in responses:
                        rule_responses[r-1] += 1
                        num_rule_evals += 1
                elif question == type and agent_type == "static-neural":
                    for r in responses:
                        static_responses[r-1] += 1
                        num_static_evals += 1
                elif question == type and agent_type == "dynamic-neural":
                    for r in responses:
                        dynamic_responses[r-1] += 1
                        num_dynamic_evals += 1


get_question_type_percentages(part1_results[1], part1_results[0], "correct")


# Normalize
human_responses /= num_human_evals
static_responses /= num_static_evals
rule_responses /= num_rule_evals
dynamic_responses /= num_dynamic_evals

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, human_responses, width, color='r')

rects2 = ax.bar(ind + width, rule_responses, width, color='y')

rects3 = ax.bar(ind + 2*width, static_responses, width, color='b')

rects4 = ax.bar(ind + 3*width, dynamic_responses, width, color='g')

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentage')
ax.set_title('Correctness Percentages By Model')
ax.set_xticks(2*width + ind)
ax.set_xticklabels(('Bad', 'Mediocre', 'Acceptable', 'Good', 'Excellent'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Human', 'Rulebased', 'Static-Neural', 'Dynamic-Neural'), loc="upper left")

plt.savefig("correct.png")
plt.show()

