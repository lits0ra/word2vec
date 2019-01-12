import sys
import MeCab
import io
from html.parser import HTMLParser
import json

mecab = MeCab.Tagger ("-Owakati")

save_word_and_id = {}

class MyHtmlStripper(HTMLParser):
    def __init__(self, s):
        super().__init__()
        self.sio = io.StringIO()
        self.feed(s)

    def handle_data(self, data):
        self.sio.write(data)

    @property
    def value(self):
        return self.sio.getvalue()

i = 1

file_num = 1
for line in open('./output.txt', 'r'):
    strip_line = MyHtmlStripper(line).value
    words = mecab.parse(strip_line)
    words = words.rstrip('\n')
    words = words.split(" ")
    for word in words:
        if word not in save_word_and_id.values():
            save_word_and_id[word] = i
            i += 1
    if i % 5000 == 0:
        print("save...")
        file_name = './outputs/output{0}.json'.format(file_num)
        file_num += 1
        with open(file_name, 'w') as f:
            json.dump(save_word_and_id, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        save_word_and_id = {}
print("end")
