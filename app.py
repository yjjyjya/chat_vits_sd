# pyopenjtalk 的安装
# https://www.bilibili.com/video/BV13t4y1V7DV/?vd_source=f17bac2fc1c6cdda2557b1601f2c6413


from flask import Flask, render_template,request,jsonify,send_file
from flask_socketio import SocketIO, emit
import openai
import json
import os
import subprocess
from scipy.io import wavfile

import matplotlib.pyplot as plt
import IPython.display as ipd

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


import sys
sys.path.append(os.path.dirname(__file__) + '\\vits')
print(sys.path)

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm




app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

openai.api_key = "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"

def get_audio(text):
    # 加载模型
    hps = utils.get_hparams_from_file(r'./vits/dlmodel/config.json')
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    utils.load_checkpoint(r"./vits/dlmodel/chisato.pth", net_g, None)
    _ = net_g.eval().to(torch.device('cpu'))

    # 生成音频文件
    wav_file = 'audio.wav'
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(torch.device('cpu')).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(torch.device('cpu'))
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    wavfile.write(wav_file, rate=hps.data.sampling_rate, data=audio)
    return audio

@app.route('/chat', methods=['POST'])
def chat():
  # 解析请求数据
  request_data = request.get_json()
  user_message = request_data['messages'][0]['content']
  # 调用 OpenAI API 进行聊天
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": user_message},
    ],
    temperature=0.7,
  )
  
  # 生成语音
  audio_file = get_audio(response.choices[0].text)

  # 发送聊天结果和语音给客户端
  # socketio.emit('chat_response', {'text': response.choices[0].text, 'audio_file': audio_file})
  socketio.emit('chat_response', response.choices[0].message['content'] )

  # 返回聊天结果
  # return jsonify({'choices': response.choices})

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5001)