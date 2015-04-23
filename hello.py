import os

from flask import Flask, request
from analyze_image import classify_image_url

app = Flask(__name__)

# added comment
@app.route('/')
def hello():
  return 'Hello World!'

@app.route('/data')
def get_image_classification():
  img_url = request.args.get('img_url')
  messages = classify_image_url(str(img_url))
  return messages
