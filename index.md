---
layout: page
title: CoCoA 
tagline: Collaborative Communicating Agents
homepage: true
---

### About 

We build **Co**llaborative **Co**mmunicating **A**gents (CoCoA) that
collaborate with humans through natural language communication.
In a *symmetric collaborative dialogue* setting, two agents,
each with private knowledge, must communicate to achieve a common goal. 

## Setting and Framework

In the symmetric collaborative dialogue setting, each dialogue is centered around a **scenario**,
which provides private knowledge to each agent and optional shared knowledge available to both agents.
Two agents communicate using dialogue to achieve a common goal and complete a **task** based on 
the information in the scenario. 

Our CoCoA framework provides tools to:

1. Create scenarios for new task types in this setting
2. Use Mechanical Turk to collect human-human dialogues centered around generated scenarios
3. Train bots that can participate in dialogues with humans, using the collected data
4. Deploy the trained bots and test their effectiveness against humans on Mechanical Turk (including by collecting reviews from Turkers)

### MutualFriends Task
We used the CoCoA framework to create a **MutualFriends** task.
In each **scenario** in this task, two agents, A and B, each have a private knowledge base (KB).
A KB contains a list of items (friends) with a value for each attribute (e.g., name, school, major, etc.).
Here is an example dialogue (given KB of agent A):

| Name        | School       | Major                | Company     |
|-------------|--------------|----------------------|-------------|
| Jessica     | Columbia     | Computer Science     | Google      |
| Josh        | Columbia     | Linguistics          | Google      |
| ...         | ...          | ...                  | ...         |

<br/>
A: Hi! Most of my friends work for Google<br/>
B: do you have anyone who went to columbia?<br/> 
A: Hello?<br/>
A: I have Jessica a friend of mine<br/>
A: and Josh, both went to columbia<br/>
B: or anyone working at apple?<br/>
B: SELECT (Jessica, Columbia, Computer Science, Google)<br/>
A: SELECT (Jessica, Columbia, Computer Science, Google)<br/>

We used the CoCoA framework to collect 11K human-human dialogues from Mechanical Turk. Human-human dialogues in this setting exhibit challenging lexical, semantic, and strategic elements,
including cross-talk, coreference, correction, coordination, implicature, and so on. We then trained a **dyn**amic knowledge graph neural **net**work (DynoNet) on the collected dialogues. We deployed conversational bots using this underlying model and asked Turkers on Mechanical Turk to chat with them to complete the MutualFriends task, and asked them to rate the bots.

### Resources

We collected 11K human-human dialogues through Amazon Mechanical Turk.<br/>
[Download](https://worksheets.codalab.org/bundles/0x5a4cefea7fd443cea15aa532bb8fcd67/) the dataset.<br/>
[Browse](https://worksheets.codalab.org/rest/bundles/0x2b7d7cb170b0475fa998f3ddf3c32893/contents/blob/chat_viewer/chat.html) the dataset.<br/>

[Code](https://github.com/stanfordnlp/cocoa/tree/cocoa-0.1)
