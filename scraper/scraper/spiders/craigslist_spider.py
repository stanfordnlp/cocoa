from scrapy import Spider, Request
from scrapy.selector import Selector
import glob
import os.path
from itertools import izip

class CraigslistSpider(Spider):
    name = 'craigslist'
    allowed_domains = ['craigslist.org']
    category_map = {
            'car': 'cto',
            'furniture': 'fuo',
            'housing': 'apa',
            'bike': 'bik',
            'phone': 'mob',
            'electronics': 'ele',
            }

    def __init__(self, num_result_pages=1, num_item_per_page=120, cache_dir=None, category='car', from_cache=False, image=0, *args, **kwargs):
        super(CraigslistSpider, self).__init__(*args, **kwargs)
        self.num_result_pages = int(num_result_pages)
        self.num_item_per_page = int(num_item_per_page)
        self.category = category
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = os.path.join(cache_dir, 'htmls', self.category)
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir)
        self.from_cache = from_cache == 'True'
        try:
            self.url_cat = self.category_map[category]
        except KeyError:
            raise ValueError('Unknown category %s' % category)
        # 0 = don't download image; 1 = just download the main/first image; 2 = download all images
        self.image = int(image)
        self.base_url = 'https://sfbay.craigslist.org/search/eby'

    def start_requests(self):
        if self.from_cache and self.cache_dir is not None:
            files = glob.glob('%s/html/*/*.html' % self.cache_dir)
            urls = ['file://%s' % x for x in files]
            for url in urls:
                yield Request(url, callback=self.parse_item_page)
        else:
            # NOTE: eby='east bay' (search area)
            url = 'https://sfbay.craigslist.org/search/eby/{category}'.format(category=self.url_cat)
            yield Request(url, callback=self.parse)

    def parse(self, response):
        urls = []
        for i in xrange(self.num_result_pages):
            urls.append(response.urljoin('?s=%d' % (i * self.num_item_per_page)))
        # Search result
        for url in urls:
            yield Request(url, callback=self.parse_result_page)

    def parse_result_page(self, response):
        ids = response.xpath('//*[@class="result-title hdrlnk"]/@data-id').extract()
        for id_ in ids:
            item_page = 'https://sfbay.craigslist.org/eby/{category}/{id}.html'.format(category=self.url_cat, id=str(id_))
            yield Request(item_page, callback=self.parse_item_page)

    def parse_item_page(self, response):
        try:
            post_id = response.url.split('/')[-1].replace('.html', '')
            title = response.xpath('//*[@id="titletextonly"]/text()').extract()[0]
            price = int(response.xpath('//*[@class="price"]/text()').re(r'\$(\d*)')[0])
            item = response.xpath('//*[@class="attrgroup"][1]/span/b/text()').extract()[0]
            attr_names = response.xpath('//*[@class="attrgroup"][2]/span/text()').re(r'(.*): ')
            attr_values = response.xpath('//*[@class="attrgroup"][2]/span/b/text()').extract()
            assert len(attr_names) == len(attr_values)
            attrs = {k: v for k, v in izip(attr_names, attr_values)}
            text = response.xpath('//*[@id="postingbody"]/text()').extract()
            processed = []
            for line in text:
                line = line.strip()
                if line:
                    processed.append(line)
            image_urls = response.xpath('//div[@id="thumbs"]/*[@data-imgid]/@href').extract()
        except (IndexError, AssertionError) as e:
            item = None

        if item is None:
            yield {}
        else:
            if self.image == 0:
                image_urls = []
            elif self.image == 1:
                image_urls = image_urls[:1]

            item = {
                    'category': self.category,
                    'post_id': post_id,
                    'title': title,
                    'item': item,
                    'price': price,
                    'attrs': attrs,
                    'description': processed,
                    'image_urls': image_urls,
                    }

            item['images'] = []

            if self.cache_dir:
                # Save html
                path = os.path.join(self.cache_dir, '%s.html' % item['post_id'])
                if not os.path.exists(path):
                    with open(path, 'w') as fout:
                        fout.write(response.body)
            yield item
