---
layout: page
title: CoCoA 
tagline: Collaborative Communicating Agents
homepage: true
---

We build **Co**llaborative **Co**mmunicating **A**gents (CoCoA) that
collaborate with humans to achieve a common goal through natural language communication.
Each dialogue is centered around a *scenario*,
which provides private knowledge to each agent and optional shared knowledge available to both agents.
Two agents communicate using dialogue to achieve a common goal and complete a task based on 
information in the scenario. 

Broadly, many dialogue tasks fall into this framework, for example,

* restaurant/movie searching where the scenario contains the user preference list and a database;
* visually grounded dialogue where the scenario contains a shared scene (image); 
* negotiation where the scenario contains terms or issues to discuss.

<br>
### Projects 

<div class="project" id="craigslist" markdown="1">

<div class="project-header">CraigslistBargain</div>

He He, Derek Chen, Anusha Balakrishnan, Percy Liang.
[Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637).
EMNLP 2018.
<button type="button" class="btn btn-outline-danger btn-sm">New!</button>

[[Download data]](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/)
[[Browse]](https://cs.stanford.edu/~hehe/transcripts.html)
[[Github Code]](https://github.com/stanfordnlp/cocoa/tree/master)


The CraigslistBargain task is designed to focus on more realistic scenarios that
invites richer language but at the same time requires strategic decision-making. 
Here, we have two agents negotiate the price of an item for sale on Craigslist.

<div class="card" markdown="1">
<div class="card-header">Example dialogue</div>
<div class="card-body" markdown="1">
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

</div>

<div class="project" id="mutualfriend" markdown="1">

<div class="project-header">MutualFriends</div>

He He, Anusha Balakrishnan, Mihail Eric, Percy Liang.
[Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/pdf/1704.07130.pdf).
ACL 2017.

[[Download data]](https://worksheets.codalab.org/bundles/0x5a4cefea7fd443cea15aa532bb8fcd67/)
[[Browse]](https://worksheets.codalab.org/rest/bundles/0x2b7d7cb170b0475fa998f3ddf3c32893/contents/blob/chat_viewer/chat.html)
[[Codalab worksheet]](https://worksheets.codalab.org/worksheets/0xc757f29f5c794e5eb7bfa8ca9c945573/)
[[Github Code]](https://github.com/stanfordnlp/cocoa/tree/mutualfriends)

Our goal is to build systems that collaborate with people
by exchanging information through natural language
and reasoning over structured knowledge base.
In the MutualFriend task, two agents, A and B, each have a *private* knowledge base,
which contains a list of friends with multiple attributes (e.g., name, school, major, etc.).
The agents must chat with each other to find their unique mutual friend.

<div class="card" markdown="1">
<div class="card-header">Example dialogue</div>
<div class="card-body" markdown="1">
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

</div>
