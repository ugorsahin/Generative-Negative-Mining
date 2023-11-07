"""
This script is the wrapper for llm interaction.
It contains utility functions.
"""

import argparse
import json
import logging
import os
import re
import time

import pandas as pd
import selenium.common.exceptions as Exceptions
from talkingheads import HuggingChatClient
from tqdm import tqdm

input_query = '''Portray each of the given words in four different ways.
To help you, I will provide a sentence the word is used, so that you can portray them relevant to context.
If the word is an animal name, replace it with other animals.
Each time, change the word with a word semantically similar in one portrayal.
Provide 2-3 words description of the portrayal in parenthesis, use visually observable words.
Provice color information in both long and short description.
Do not write explanations. Only list the portrayals.

Example:
Words: bird, rock
Sentence: a bird on a rock in a body of water

Answer:
Bird:
1) A majestic bald eagle perched on a tree branch (bald eagle)
2) A group of pigeons pecking at bread crumbs on the sidewalk (pigeons)
3) A colorful kite with purple and blue colors (blue purple kite)
4) A seagull squawking and fighting over a fish (seagulls)

Rock:
1) A jagged, gray granite boulder by the shoreline (gray granite boulder)
2) A smooth, rounded pebble glistening in the sunlight (glistening pebble)
3) A blue rock with green stripes (blue-green rock)
4) A stack of neatly arranged flat stones forming a cairn (cairn of stones)

Confirm you understand the next and we will start'''

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%y/%m/%d - %H:%M:%S',
    level=logging.INFO
)

def remove_unseen(words, sentence):
    existent_words = list(filter(
        lambda x: x in sentence,
        words)
    )
    return existent_words

def save_portrayal(_portre, path_to_file):
    pd.DataFrame(_portre).to_json(path_to_file, orient='records')

def parse_portrayal(raw_text, items):
    items_regex = r':?\n((?:.+\n){4})'.join(list(items) + [''])
    results = re.search(items_regex.lower(), raw_text.lower() + '\n') or {}
    if results:
        results = {
            items[idx] : i.split('\n')
            for idx, i in enumerate(results.groups())
        }
    return results

def process_one_item(chathead, item):
    words = ', '.join(item['items'])
    _query = f'Words: {words}\nSentence: {item["tag2text_caption"]}'
    response = ''

    response = chathead.interact(_query)
    logging.info(response)
    parsed_response = parse_portrayal(response, item['items'])
    # print(parsed_response)

    return response, parsed_response

def main(args):
    logging.info('loading input file')
    tag2text_output = pd.read_json(args.input_path).to_dict(orient='records')
    logging.info('loaded input file')

    portrayals = {}
    if os.path.exists(args.save_path):
        logging.info('loading %s', args.save_path)
        with open(args.save_path, 'r', encoding='utf-8') as fd:
            portrayals = json.loads(fd.read())

    logging.info('initializing chathead')
    chathead = HuggingChatClient(
        username=args.username,
        password=args.password,
        headless=False,
        verbose=True
    )
    answer = chathead.interact(input_query)
    logging.info('chathead is ready')
    logging.info(answer)

    consecutive_count = 0

    for an_image in tqdm(tag2text_output):
        im_name = an_image['save_path']
        if portrayals.get(im_name) is not None:
            continue
        if consecutive_count >= args.save_period:
            chathead.reset_thread()
            _ = chathead.interact(input_query)
            consecutive_count = 0

        try:
            response, parsed_response = process_one_item(chathead, an_image)
        except Exceptions.NoSuchElementException:
            print('Error occured, probably rate-limiting')
            save_portrayal(portrayals, args.save_path)
            time.sleep(args.wait_on_rate_limit)
            chathead.reset_thread()
            _ = chathead.interact(input_query)
            continue

        portrayals[im_name] = {**an_image, 'llm_output' : response, 'variations': parsed_response}

        if len(portrayals) % args.save_period == 0:
            save_portrayal(portrayals, args.save_path)

        consecutive_count += 1

    save_portrayal(portrayals, args.save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--input_path", default='../outputs/tag2text.json')
    parser.add_argument("--save_path", default='../outputs/portrayals.json')
    parser.add_argument("--wait_on_rate_limit", default=3600)
    parser.add_argument("--save_period", default=1, type=int)
    args = parser.parse_args()

    main(args)
