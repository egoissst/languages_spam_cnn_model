import urllib.request, json 
import csv

csv.field_size_limit(100000000)

def getJsonFromUrl(url):
    with urllib.request.urlopen(finalUrl) as url:
        data = json.loads(url.read().decode())
        return data

def getListFromCsv(filePath):
    with open(filePath, 'r', encoding='utf-8') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
        return lines

def getListFromCsvV2(filePath):
    with open(filePath, 'r', encoding='utf-8') as readFile:
        lines = [line.rstrip('\n') for line in readFile]
        return lines


def get_file_string_from_config_props(FLAGS, cfg):
    """
    returns a string to be written to a file for information on the config
    params - 
    FLAGS - tf.FLAGS
    cfg - config loaded from yaml
    """
    first_property = 'dev_sample_percentage'
    a = [(x, FLAGS[x].value) for x in FLAGS]
    index_start = [item[0] for item in a].index(first_property)
    prop_dict = a[index_start:]

    prop_dict.append(None)
    prop_dict.append(None)

    w2v_props = cfg["word_embeddings"]["word2vec"]

    for item in w2v_props:
        prop_dict.append(('w2v_' + item, w2v_props[item]))

    prop_dict.append(None)
    prop_dict.append(None)

    prop_dict.append(('positive_data_file', cfg["datasets"]["mrpolarity"]["positive_data_file"]["path"]))
    prop_dict.append(('negative_data_file', cfg["datasets"]["mrpolarity"]["negative_data_file"]["path"]))

    prop_dict.append(None)

    list_lines = list()
    for item in prop_dict:
        list_lines.append("{:30} : {}".format(item[0], item[1]) if item else "")


    file_str = '\n'.join(list_lines)
    return file_str
