# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


import sqlite3

class InformedSlpPipeline:
    def __init__(self):
        self.conn = sqlite3.connect('informedslp.db')
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS articles 
                         (title text, link text, content text, authors text, citations text)''')

    def process_item(self, item, spider):
        self.c.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?)", 
            (item['title'], item['link'], item['content'], item['authors'], item['citations']))
        self.conn.commit()
        return item