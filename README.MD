# Nudity image detection HTTP API

This project provides a ready to deploy REST API allowing to predict if an image is offensive or has adult content.

It relies on [open_nsfw](https://github.com/yahoo/open_nsfw), which provides a pre-trained open source neural network model for [Caffe](https://github.com/BVLC/caffe).

The current project doesn't aim at improving the quality of the predictions. The main goal is to provide a ready to deploy
solution for people that might need this kind of service for free.

### Running and Deploying

The python REST API, with Caffe and open_nsfw are packaged all together in a Docker image. So in theory you could just 
use an [existing build from Docker Hub](https://hub.docker.com/r/eugencepoi/nsfw_api/) or build your own.

#### Running locally

The API and everything needed for it to work is part of a docker image. So first you will need to install Docker for your OS.
Then you can use a prebuilt image.

```docker run -it -p 127.0.0.1:5000:5000/tcp --env PORT=5000 eugencepoi/nsfw_api:latest```

The API should be up and running at

```
http://localhost:5000/?url=https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png`
```

Or if you want to tweak things around you can clone the repository and then build the image your self.
Build it with `docker-compose build` and use `docker-compose up` to start the container.

#### Running on Heroku

A simple and quick option to get it up and ready for you to use from your website is via Heroku. You will need an Heroku 
account for that. If you are building your own image from a clone of this repository, then from inside the project directory do:

 - Login to the Heroku container registry `heroku container:login`
 - Create an app if one doesn't already exist `heroku create YOUR_APP_NAME`
 - Upload the image `heroku container:push web` (add optionally `--app YOUR_APP_NAME`)
 - Release it to your app `heroku container:release web` (add optionally `--app YOUR_APP_NAME`)

The service should be up and running. It might take a bit of time for the first request to be processed, though subsequent ones
should be faster.


#### Running in offline mode

For now running in batch/offline mode is outside of the scope of the project but any contribution to do so is welcome.


### API Usage

You can use the classification API via two endpoints.

#### Single prediction

To get a prediction score for one single image just use the GET endpoint with a parameter named `url`.

```
curl -X GET -H 'Content-Type: application/json' http://localhost:5000\?url\=https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png

{
  "score": 0.00016061133646871895,
  "url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
}
```

The response will be a json object with a `score` property having a value between 0 and 1, 1 meaning it is for sure
adult content while 0 it isn't. If there is an error while fetching the URL there will be two properties `error_code` 
and `error_reason` instead of the `score`:

```
{
  "error_code": 500,
  "error_reason": "[Errno -2] Name or service not known",
  "url": "https://foobar"
}
```

Remark that doing the classification isn't a fast operation so you shouldn't call this API in places where you want the 
response in real time/low latency (for ex. to display it to the user), but instead call the API periodically for a batch
of images using the endpoint below.

#### Batch predictions with streamed responses

The batch endpoint takes as input a list of images to classify and returns the result for each image. The response is 
being streamed back, so you could read it in a streaming fashion and process the results as they come in (as opposed 
to wait for the entire response before processing it). For that purpose use the batch classification API as follows.

```
curl -X POST -H 'Content-Type: application/json' \
 -d '{"images": [{"url": "http://foo.bar", "id": 1}, {"url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "id": 2, "extra_props": {"foo": "bar"}}]}' \
 http://localhost:5000/batch-classify
 
{"predictions": [
{"url": "http://foo.bar", "error_reason": "[Errno -2] Name or service not known", "error_code": 500, "id": 1},
{"url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "score": 0.00016061133646871895, "id": 2, "extra_props": {"foo": "bar"}}
]}
```

Each entry should have an `url` property pointing to the image accessible over HTTP/HTTPS. Any extra attribute will be 
preserved and sent back in the response. This allows easily to identify each entry and eventually pass along some context.

Failing to process one entry in the batch will not fail the entire operation, instead the result for this single
entry will report the error. However if there is an error in handling the input JSON or some other global error the format
of the response is not guaranteed so you should check the response status code.


### License
Code licensed under the [BSD 2 clause license] (https://github.com/BVLC/caffe/blob/master/LICENSE). See LICENSE file for terms.

