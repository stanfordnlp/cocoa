from lxml import html
import requests
import random
import argparse
import sys
import json

def scrape(link, xpath, n):
    page = requests.get(link)
    tree = html.fromstring(page.content)
    nodes = tree.xpath(xpath)
    strings = [node.text.strip() for node in nodes if node.text]
    return random.sample(strings, min(n, len(strings)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-schools', type=int, default=200)
    parser.add_argument('--num-names', type=int, default=200)
    parser.add_argument('--num-majors', type=int, default=200)
    parser.add_argument('--schema-path')
    args = parser.parse_args()
    random.seed(0)

    # Names
    p = ".//*[@id='content']/table/tr/td[2]/table/tbody/tr[position()<last()]/td"
    xpath = "%s[%d] | %s[%d]" % (p, 2, p, 4)
    names = scrape('https://www.ssa.gov/oact/babynames/decades/century.html', xpath, args.num_names)
    print '%d names' % len(names)

    # Schools
    schools = scrape('http://doors.stanford.edu/~sr/universities.html', '//body/ol/li/a', args.num_schools)
    print '%d schools' % len(schools)

    # Majors
    majors = scrape('http://www.utdallas.edu/academics/majors.html', ".//*[@id='programs']/tr/td[1]/a", args.num_majors)
    majors = [m for m in majors if len(m.split()) <= 2]
    print '%d majors' % len(majors)

    # Companies
    companies = ['Apple', 'Amazon', 'Alibaba', 'Airbnb', 'Google', 'Microsoft', 'Facebook', 'Pinterest', 'Pixar', 'Citibank', 'Goldman Sachs', 'Sony', 'Samsung', 'Volkswagen', 'Ford', 'Honda', 'New York Times', 'Washington Post', 'CNN']
    print '%d companies' % len(companies)

    # Schema
    schema = {
        'values': {
            'name': names,
            'school': schools,
            'major': majors,
            'company': companies,
            },
        'attributes': [
            {"name": "Name", "value_type": "name", "unique": False},
            {"name": "Company", "value_type": "company", "unique": False},
            {"name": "Major", "value_type": "major", "unique": False},
            {"name": "Hobby", "value_type": "hobby", "unique": False}
            ]
        }
    with open(args.schema_path, 'w') as out:
        #json.dump(schema, out)
        print >>out, json.dumps(schema, indent=2, separators=(',', ':'))
