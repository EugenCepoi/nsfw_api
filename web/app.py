from flask import Flask, request, jsonify
import urllib2
import caffe
import contextlib
import collections

import classifier

app = Flask(__name__)

ClassificationResult = collections.namedtuple('ClassificationResult', 'url score')


@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    json = request.get_json(force=True)

    if "urls" in json:
        image_entries = list(map(lambda u: {'url': u}, json["urls"]))
    else:
        image_entries = json["images"]

    return jsonify(predictions=classify_from_urls(image_entries))


@app.route('/')
def single_classify():
    single_image = {'url': request.args.get('url')}
    result = classify_from_urls([single_image])[0]
    return jsonify(result)


def classify_from_urls(image_entries):
    nsfw_net = caffe.Net(
        "/opt/nsfw_model/deploy.prototxt",
        "/opt/nsfw_model/resnet_50_1by2_nsfw.caffemodel",
        caffe.TEST
    )

    return list(map(lambda e: classify_from_url(e, nsfw_net), image_entries))


def classify_from_url(image_entry, nsfw_net):
    headers = {'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'}
    req = urllib2.Request(image_entry["url"], None, headers)

    with contextlib.closing(urllib2.urlopen(req)) as stream:
        score = classifier.classify(stream.read(), nsfw_net)
        result = {'score': score}
        result.update(image_entry)
        return result


if __name__ == '__main__':
    # TODO port number env var?
    app.run(host='0.0.0.0')

