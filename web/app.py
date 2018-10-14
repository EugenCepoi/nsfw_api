from flask import Flask, request, Response, jsonify
import json
import urllib2
import caffe
import contextlib
import numpy as np
import classify_nsfw


def make_transformer(nsfw_net):
    # Load transformer
    # Note that the parameters are hard-coded for best results
    transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer


nsfw_net = caffe.Net(
    "/opt/open_nsfw/nsfw_model/deploy.prototxt",
    "/opt/open_nsfw/nsfw_model/resnet_50_1by2_nsfw.caffemodel",
    caffe.TEST
)
caffe_transformer = make_transformer(nsfw_net)
app = Flask(__name__)


@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    req_json = request.get_json(force=True)

    if "urls" in req_json:
        image_entries = list(map(lambda u: {'url': u}, req_json["urls"]))
    elif "images" in req_json:
        image_entries = req_json["images"]
    else:
        return 'Accepted formats are {"urls": ["url1", "url2"]} or {"images": [{"url":"url1"}, {"url":"url2"}]}'

    def stream_predictions():
        predictions = classify_from_urls(image_entries).__iter__()
        try:
            prev_prediction = next(predictions)
        except StopIteration:
            yield '{"predictions": []}'
            raise StopIteration
        yield '{"predictions": [\n'
        for prediction in predictions:
            yield json.dumps(prev_prediction) + ',\n'
            prev_prediction = prediction
        yield json.dumps(prev_prediction) + '\n]}'

    return Response(stream_predictions(), mimetype='application/json')


@app.route('/')
def single_classify():
    if request.args.has_key('url'):
        single_image = {'url': request.args.get('url')}
        result = classify_from_urls([single_image]).next()
        return jsonify(result)
    else:
        return "Missing  url parameter", 400


def classify_from_urls(image_entries):
    for e in image_entries:
        yield classify_from_url(e, nsfw_net)


def classify_from_url(image_entry, nsfw_net):
    # Otherwise it seems that public S3 buckets are not accessible through HTTP(s)
    headers = {'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'}

    try:
        req = urllib2.Request(image_entry["url"], None, headers)
        with contextlib.closing(urllib2.urlopen(req)) as stream:
            score = classify(stream.read(), nsfw_net)
            result = {'score': score}
    except urllib2.HTTPError, e:
        result = {'error_code': e.code, 'error_reason': e.reason}
    except urllib2.URLError, e:
        result = {'error_code': 500, 'error_reason': str(e.reason)}
    except Exception, e:
        result = {'error_code': 500, 'error_reason': e.message}


    result.update(image_entry)
    return result


def classify(image_data, nsfw_net):
    # Classify.
    scores = classify_nsfw.caffe_preprocess_and_compute(
        image_data,
        caffe_transformer=caffe_transformer,
        caffe_net=nsfw_net,
        output_layers=['prob']
    )

    return scores[1]


if __name__ == '__main__':
    app.run(host='0.0.0.0')

