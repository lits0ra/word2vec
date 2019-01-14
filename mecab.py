import sys
import MeCab
import io
from html.parser import HTMLParser
import json, time

mecab = MeCab.Tagger ("-Owakati")

save_word_and_id = {}
word_and_id = {}

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
        if word not in word_and_id.keys():
            save_word_and_id[word] = i
            word_and_id[word] = i
            i += 1
    if i % 5000 == 0:
        file_name = './outputs/output{0}.json'.format(file_num)
        if save_word_and_id:
            file_num += 1
            with open(file_name, 'w') as f:
                json.dump(save_word_and_id, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
            save_word_and_id = {}
print("end")
