import pytest
from model.graph_embedder import GraphEmbedder, GraphEmbedderConfig
from model.graph import Graph, GraphMetadata, GraphBatch
from basic.schema import Schema
from basic.lexicon import Lexicon
from model.preprocess import Preprocessor, build_schema_mappings
from basic.kb import KB

max_degree = 5

@pytest.fixture(scope='session')
def config(metadata):
    num_edge_labels = metadata.relation_map.size
    node_embed_size = 4
    edge_embed_size = 4
    utterance_size = metadata.utterance_size
    feat_size = metadata.feat_size
    batch_size = 2
    return GraphEmbedderConfig(num_edge_labels, node_embed_size, edge_embed_size, utterance_size, feat_size, batch_size=batch_size, max_degree=max_degree)

@pytest.fixture(scope='session')
def graph_embedder(config):
    return GraphEmbedder(config)

@pytest.fixture(scope='session')
def tokens():
    u1 = [('alice', ('alice', 'person')), 'works', 'at', ('google', ('google', 'company'))]
    u2 = [('ian', ('ian', 'person')), 'likes', ('hiking', ('hiking', 'hobby'))]
    return [u1, u2]

@pytest.fixture(scope='session')
def schema():
    return Schema('data/friends-schema.json')

@pytest.fixture(scope='session')
def lexicon(schema):
    return Lexicon(schema, False)

@pytest.fixture(scope='session')
def metadata(schema):
    num_items = 4
    entity_map, relation_map = build_schema_mappings(schema, num_items)
    return GraphMetadata(schema, entity_map, relation_map, 3, 15, max_degree=max_degree)

@pytest.fixture
def graph(metadata, schema):
    Graph.metadata = metadata
    items = [{'Name': 'Alice', 'Company': 'Microsoft', 'Hobby': 'hiking'},\
             {'Name': 'Bob', 'Company': 'Apple', 'Hobby': 'hiking'}]
    kb = KB.from_dict(schema, items)
    return Graph(kb)

@pytest.fixture
def graph2(schema):
    items = [{'Name': 'Alice', 'Company': 'Microsoft', 'Hobby': 'reading'},\
             {'Name': 'Bob', 'Company': 'Apple', 'Hobby': 'hiking'}]
    kb = KB.from_dict(schema, items)
    return Graph(kb)

@pytest.fixture
def graph_batch(graph, graph2):
    return GraphBatch((graph, graph2))

