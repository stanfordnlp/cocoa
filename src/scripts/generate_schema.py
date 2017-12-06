from lxml import html
import requests
import argparse
import json
import re
import os

def clean(s):
    s = re.sub(r'\([^)]*\)', '', s)
    return s.strip()

def scrape(link, xpath, cache):
    cached_file = '%s/%s' % (cache, link.split('/')[-1])
    if os.path.isfile(cached_file):
        with open(cached_file, 'r') as fin:
            content = fin.read()
    else:
        content = page = requests.get(link).content
        with open(cached_file, 'w') as fout:
            fout.write(content)
    tree = html.fromstring(content)
    nodes = tree.xpath(xpath)
    strings = [clean(node.text) for node in nodes if node.text]
    return [s for s in strings if not isinstance(s, unicode)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path')
    parser.add_argument('--cache-path', default='data/cache')
    args = parser.parse_args()

    # Names
    p = ".//*[@id='content']/div/table/tr/td[2]/table/tbody/tr[position()<last()]/td"
    xpath = "%s[%d] | %s[%d]" % (p, 2, p, 4)
    names = scrape('https://www.ssa.gov/oact/babynames/decades/century.html', xpath, args.cache_path)
    print '%d names' % len(names)

    # Schools
    schools = scrape('http://doors.stanford.edu/~sr/universities.html', '//body/ol/li/a', args.cache_path)
    print '%d schools' % len(schools)

    # Majors
    xpath = "//body/div/table/td/a | //body/div/table/tr/td/a"
    majors = scrape('http://www.a2zcolleges.com/majors', xpath, args.cache_path)
    majors = [re.sub(r'/[^/ ]*', '', major) for major in majors]
    print '%d majors' % len(majors)

    # Companies
    companies = scrape('https://en.wikipedia.org/wiki/List_of_companies_of_the_United_States', ".//*[@id='mw-content-text']/div/ul/li/a", args.cache_path)
    print '%d companies' % len(companies)

    # Hobbies
    hobbies = scrape('https://en.wikipedia.org/wiki/List_of_hobbies', ".//*[@id='mw-content-text']/div/ul/li/a", args.cache_path)
    print '%d hobbies' % len(hobbies)

    # Location preference
    loc_pref = ['indoor', 'outdoor']

    # Time preference
    time_pref = ['morning', 'afternoon', 'evening']

    # Schema
    schema = {
        'values': {
            'name': names,
            'school': schools,
            'major': majors,
            'company': companies,
            'hobby': hobbies,
            'time_pref': time_pref,
            'loc_pref': loc_pref
            },
        'attributes': [
            {"name": "Name", "value_type": "name", "unique": False},
            {"name": "School", "value_type": "school", "unique": False},
            {"name": "Major", "value_type": "major", "unique": False},
            {"name": "Company", "value_type": "company", "unique": False},
            {"name": "Hobby", "value_type": "hobby", "unique": False},
            {"name": "Time Preference", "value_type": "time_pref", "unique": False},
            {"name": "Location Preference", "value_type": "loc_pref", "unique": False}
            ]
        }
    with open(args.schema_path, 'w') as out:
        #json.dump(schema, out)
        print >>out, json.dumps(schema, indent=2, separators=(',', ':'))
