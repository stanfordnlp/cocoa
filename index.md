---
layout: page
title: CoCoA 
tagline: Collaborative Communicating Agents
homepage: true
---

We build **Co**llaborative **Co**mmunicating **A**gents (CoCoA) that
collaborate with humans to achieve a common goal through natural language communication.
Our main focus is task-oriented dialogue where two agents converse with each other to complete a task in a grounded scenario.
We provide a platform that facilitates research in dialogue systems from data collection to evaluation.

## Framework

In the collaborative dialogue setting, each dialogue is centered around a **scenario**,
which provides private knowledge to each agent and optional shared knowledge available to both agents.
Two agents communicate using dialogue to achieve a common goal and complete a **task** based on 
the information in the scenario. 

Our CoCoA framework provides tools to:

1. Create scenarios for new tasks in this setting;
2. Use Amazon Mechanical Turk (AMT) to collect human-human dialogues centered around generated scenarios;
3. Train bots that can converse with humans to complete the task, using the collected data;
4. Deploy the trained bots and test their effectiveness against humans on AMT (including collecting reviews from Turkers).

Broadly, many dialogue tasks fall into this framework, for example,

* restaurant/movie searching where the scenario contains the user preference list and a database;
* visually grounded dialogue where the scenario contains a shared scene (image); 
* negotiation where the scenario contains terms or issues to discuss.

## Projects 

### MutualFriends

He He, Anusha Balakrishnan, Mihail Eric, Percy Liang.
[Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings]().
ACL 2017.

We design a **MutualFriends** task in the CoCoA framework.
Two agents, A and B, each have a private knowledge base (KB),
which contains a list of items (friends) with a value for each attribute (e.g., name, school, major, etc.).
The goal is to chat with each other to find the unique mutual friend.
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

We collected 11K human-human dialogues from AMT, which exhibit challenging lexical, semantic, and strategic elements,
including cross-talk, coreference, correction, coordination, implicature, and so on.
Further, we proposed a **dy**amic k**no**wledge graph **net**work (DynoNet) that incorporate both structured knowledge and unstructured language to represent the dialogue state.
For evaluation, we deployed the bots and paired them with Turkers to complete the MutualFriends task.
The bots were rated by their human partners in terms of fluency, correctness, cooperation, and human-likeness.

All results can be found in the Codalab [worksheet](https://worksheets.codalab.org/worksheets/0xc757f29f5c794e5eb7bfa8ca9c945573/).

## Resources

[Download](https://worksheets.codalab.org/bundles/0x5a4cefea7fd443cea15aa532bb8fcd67/)
and
[Browse](https://worksheets.codalab.org/rest/bundles/0x2b7d7cb170b0475fa998f3ddf3c32893/contents/blob/chat_viewer/chat.html)
the MutualFriends dataset.<br/>
[Code](https://github.com/stanfordnlp/cocoa)
