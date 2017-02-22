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

### Environment
We set up a **MutualFriends** task.
In this task, two agents, A and B, each has a private knowledge base (KB).
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
A: Hello?
// *Cross talk*<br/>
A: I have Jessica a friend of mine<br/>
A: and Josh, both went to columbia<br/>
B: or anyone working at apple?
// *Cross talk*<br/>
B: SELECT (Jessica, Columbia, Computer Science, Google)<br/>
A: SELECT (Jessica, Columbia, Computer Science, Google)<br/>

Human-human dialogues in this setting exhibit challenging lexical, semantic, and strategic elements,
including cross-talk, coreference, correction, coordination, implicature, and so on.

### Dataset

We have collected 11K human-human dialogues through Amazon Mechanical Turk.

[Download](https://codalab.stanford.edu/bundles/0x1a3c6bde41544416996c11fa3ffb2dfc/) the dataset (split into train, dev, test sets).

[Browse](https://codalab.stanford.edu/rest/bundles/0xc9d7f5b32f80457b9c05db63bc788f4b/contents/blob/chat.html) the dataset in html.
