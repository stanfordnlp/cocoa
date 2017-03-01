# CoCoA (Collaborative Communicating Agents)                                                    
We build Collaborative Communicating Agents (CoCoA) that collaborate with humans through natural language communication. We focus on the symmetric collaborative dialogue setting, where two agents, each with private knowledge, must communicate to achieve a common goal.

This branch contains code for the paper "Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings".

## Dependencies
The following Python packages are required.
- General: Python 2.7, NumPy 1.11, Tensorflow r0.12
- Lexicon: fuzzywuzzy, editdistance
- Web server: TODO

## Data collection
### Scenario generation
Generate the schema:
```
mkdir data/cache
python src/scripts/generate_schema.py --schema-path data/schema.json --cache-path data/cache
```
Generate scenarios from the schema:
TODO

### Set up the web server
TODO

## Model
### Training

### Evaluation 









