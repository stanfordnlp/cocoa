# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
import os.path
import logging
import scrapy
from scraper.settings import IMAGES_STORE

class ScraperPipeline(object):
    def process_item(self, item, spider):
        return item

class CraigslistValidationPipeline(object):
    def process_item(self, item, spider):
        if item == {}:
            raise DropItem('parse error')
        else:
            return item

class CraigslistImagesPipeline(ImagesPipeline):
    def get_media_requests(self, item, info):
        for i, image_url in enumerate(item['image_urls']):
            meta = {'filename': '%s/%s_%d.jpg' % (item['category'], item['post_id'], i)}
            yield scrapy.Request(image_url, meta=meta)

    def file_path(self, request, response=None, info=None):
        filename = request.meta['filename']
        return filename

    def item_completed(self, results, item, info):
        for i, result in enumerate([x for ok, x in results if ok]):
            path = result['path']  # path is relative to IMAGES_STORE
            item['images'].append(path)
        return item

