import json
from cocoa.core.util import read_json

# Global statistics that we can output to monitor the run.

stats_path = None
STATS = {}

def init(path):
    global stats_path, STATS
    stats_path = path
    try:
        STATS = read_json(stats_path)
    except Exception:
        STATS = {}

def add(*args):
    # Example: add_stats('data', 'num_examples', 3)
    s = STATS
    prefix = args[:-2]
    for k in prefix:
        if k not in s:
            s[k] = {}
        s = s[k]
    s[args[-2]] = args[-1]
    flush()

def add_args(key, args):
    add(key, dict((arg, getattr(args, arg)) for arg in vars(args)))

def update(stats):
    for k in stats:
        STATS[k] = stats[k]
    flush()

def flush():
    if stats_path:
        out = open(stats_path, 'w')
        print >>out, json.dumps(STATS)
        out.close()

############################################################

# summary: {'mean': ...}
# summary_map: {key: summary}

def summary_to_str(s):
    return '%g / %g / %g (%g)' % (s['min'], s['mean'], s['max'], s['count'])

def summary_map_to_str(m):
    return ' '.join('%s=%g' % (k, s['mean'] if isinstance(s, dict) else s) for k, s in sorted(m.items()))

def update_summary_map(m1, m2):
    for k, s in m2.items():
        if k not in m1:
            m1[k] = {}
        update_summary(m1[k], s)

def update_summary(s1, s2):
    if isinstance(s2, dict):
        s1['min'] = min(s1.get('min', s2['min']), s2['min'])
        s1['max'] = max(s1.get('max', s2['max']), s2['max'])
        s1['sum'] = s1.get('sum', 0) + s2['sum']
        s1['count'] = s1.get('count', 0) + s2['count']
    else:
        s1['min'] = min(s1.get('min', s2), s2)
        s1['max'] = max(s1.get('max', s2), s2)
        s1['sum'] = s1.get('sum', 0) + s2
        s1['count'] = s1.get('count', 0) + 1
    s1['mean'] = 1.0 * s1['sum'] / s1['count']

def dump_summary_map(m):
    for k, s in m.items():
        print k, '=', summary_to_str(s)
