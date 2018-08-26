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

### Framework

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

### Projects 

<div class="project" id="craigslist" markdown="1">

<div class="project-header">CraigslistBargain</div>

He He, Derek Chen, Anusha Balakrishnan, Percy Liang.
[Decoupling Strategy and Generation in Negotiation Dialogues]().
EMNLP 2018.

The CraigslistBargain task is designed to focus on more realistic scenarios that
invites richer language but at the same time requires strategic decision-making. 
Here, we have two agents negotiate the price of an item for sale on Craigslist.
For example, an example dialogue given the following post:

<div class="panel panel-success" markdown="1">
<div class="panel-heading">Example dialogue</div>
<div class="panel-body" markdown="1">
*JVC HD-ILA 1080P 70 Inch TV ($275)*<br/>
*Tv is approximately 10 years old. Just installed new lamp. There are 2 HDMI inputs. Works and looks like new.*
<hr/>
A: Hello <br>
B: Hello there<br>
A: So, are you interested in this great TV? Honestly, I barely used it and decided to sell it because I don't really watch much TV these days. I'm selling it for $275<br>
B: I am definitely interested in the TV, but it being 10 years old has me a bit skeptical. How does the TV look running movies and games, if you don't mind me asking.<br>
A: It's full HD at 1080p and it looks great. The TV works like it is brand new. I'll throw in a DVD player that was hooked up to it for the same price of $275<br>
B: The DVD player sounds nice, but unfortunately I'm on somewhat of a budget. Would you be willing to drop the price a tad, maybe $230?<br>
A: $230 is kind of low. I'll tell ya what, if you come pick it up where it is located I'll sell it for $260<br>
B: Throw in a couple of movies with that DVD player,and you have yourself a deal.<br>
A: Deal.<br>
B: OFFER $260.00<br>
A: ACCEPT<br>
</div>
</div>

We have collected 6.6K dialogues.
[Download](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/) and
[browse](https://cs.stanford.edu/~hehe/transcripts.html)
the dataset.

</div>

<div class="project" id="mutualfriend" markdown="1">

<div class="project-header">MutualFriends</div>

He He, Anusha Balakrishnan, Mihail Eric, Percy Liang.
[Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/pdf/1704.07130.pdf).
ACL 2017.

We design a MutualFriends task in the CoCoA framework.
Two agents, A and B, each have a private knowledge base (KB),
which contains a list of items (friends) with a value for each attribute (e.g., name, school, major, etc.).
The goal is to chat with each other to find the unique mutual friend.

<div class="panel panel-success" markdown="1">
<div class="panel-heading">Example dialogue</div>
<div class="panel-body" markdown="1">
| Name        | School       | Major                | Company     |
|-------------|--------------|----------------------|-------------|
| Jessica     | Columbia     | Computer Science     | Google      |
| Josh        | Columbia     | Linguistics          | Google      |

<br/>
A: Hi! Most of my friends work for Google<br/>
B: do you have anyone who went to columbia?<br/> 
A: Hello?<br/>
A: I have Jessica a friend of mine<br/>
A: and Josh, both went to columbia<br/>
B: or anyone working at apple?<br/>
B: SELECT (Jessica, Columbia, Computer Science, Google)<br/>
A: SELECT (Jessica, Columbia, Computer Science, Google)<br/>

</div>
</div>

We collected 11K human-human dialogues from AMT.
You can [download](https://worksheets.codalab.org/bundles/0x5a4cefea7fd443cea15aa532bb8fcd67/)
or [browse](https://worksheets.codalab.org/rest/bundles/0x2b7d7cb170b0475fa998f3ddf3c32893/contents/blob/chat_viewer/chat.html)
the dataset.
All results can be found in the Codalab [worksheet](https://worksheets.codalab.org/worksheets/0xc757f29f5c794e5eb7bfa8ca9c945573/).

</div>

### Resources
[Code](https://github.com/stanfordnlp/cocoa)
