# Necessary Imports
import cv2
from flask import Flask, redirect, render_template, request, url_for, jsonify
from werkzeug.exceptions import BadRequest
import numpy as np
import os
import pytube
import scipy.spatial.distance as ssd
import scipy.io.wavfile as wav
from scipy.signal import resample
import skimage.metrics as sm
from skimage import metrics
from python_speech_features import mfcc
import subprocess
import tempfile
import urllib.request

import librosa
import moviepy.editor as mp
from pyAudioAnalysis import audioSegmentation

import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads')
app.config['TIMEOUT'] = 180


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded videos from the form
        video1 = request.files['video1']
        video2 = request.files['video2']

        # Save the uploaded videos to the uploads folder
        video1.save(os.path.join(app.config['UPLOAD_FOLDER'], video1.filename))
        video2.save(os.path.join(app.config['UPLOAD_FOLDER'], video2.filename))

        # Redirect to the compare page with the video filenames as arguments
        return redirect(url_for('compare', video1=video1.filename, video2=video2.filename))

    return render_template('home.html')


# Downloading Video from Youtube
def download_video(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    video.download()


# Audio Analysis
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    audio.write_audiofile(audio_path, logger=None)
    return audio_path


def extract_mfcc(audio_path):
    (rate, signal) = wav.read(audio_path)
    mfcc_feat = []
    # Compute MFCCs for every 0.1 seconds of audio
    for i in range(0, len(signal), int(rate * 0.1)):
        mfcc_feat.append(mfcc(signal[i:i + int(rate * 0.1)], rate))
    return np.concatenate(mfcc_feat)


def extract_mfcc_num_speakers(audio_path):
    (rate, signal) = wav.read(audio_path)
    speaker_labels = audioSegmentation.speaker_diarization(signal, rate)
    num_speakers = len(set(speaker_labels))
    return num_speakers


def compare_audio(mfcc1, mfcc2):
    min_len = min(len(mfcc1), len(mfcc2))
    mfcc1 = mfcc1[:min_len, :]
    mfcc2 = mfcc2[:min_len, :]
    if len(mfcc1) < len(mfcc2):
        mfcc1 = resample(mfcc1, len(mfcc2), axis=0)
    elif len(mfcc2) < len(mfcc1):
        mfcc2 = resample(mfcc2, len(mfcc1), axis=0)
    return ssd.cosine(mfcc1.ravel(), mfcc2.ravel())


# Video Analysis
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 10 == 0:  # Extract every 10th frame
            frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames


def compare_frames(frames1, frames2):
    mse = np.mean([np.mean((frame1 - frame2) ** 2)
                  for frame1, frame2 in zip(frames1, frames2)])
    ssim = np.mean([sm.structural_similarity(frame1, frame2, channel_axis=-1)
                    for frame1, frame2 in zip(frames1, frames2)])
    return mse, ssim


def compute_optical_flow(frames):
    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    count = 0
    for frame in frames[1:]:
        count += 1
        if count % 5 == 0:  # Compute optical flow for every 5th frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            prev_gray = gray
    return flows


def compare_optical_flow(flows1, flows2):
    cosine_similarities = []
    for flow1, flow2 in zip(flows1, flows2):
        flat_flow1 = flow1.reshape(-1)
        flat_flow2 = flow2.reshape(-1)
        cosine_similarities.append(ssd.cosine(flat_flow1, flat_flow2))
    return np.mean(cosine_similarities)


# Calculate Similarity
def calculate_similarity(video1_path, video2_path):
    audio1_path = extract_audio(video1_path)
    audio2_path = extract_audio(video2_path)
    mfcc1 = extract_mfcc(audio1_path)
    mfcc2 = extract_mfcc(audio2_path)
    audio_similarity = compare_audio(mfcc1, mfcc2)
    frames1 = extract_frames(video1_path)
    frames2 = extract_frames(video2_path)
    frame_similarity = compare_frames(frames1, frames2)
    flows1 = compute_optical_flow(frames1)
    flows2 = compute_optical_flow(frames2)
    flow_similarity = compare_optical_flow(flows1, flows2)
    overall_similarity = (
        audio_similarity + frame_similarity[0] + frame_similarity[1] + flow_similarity) / 4
    return overall_similarity


@app.route('/compare')
def compare():
    video1 = request.args.get('video1')
    video2 = request.args.get('video2')
    video1_path = os.path.join(app.config['UPLOAD_FOLDER'], video1)
    video2_path = os.path.join(app.config['UPLOAD_FOLDER'], video2)

    # Check if both videos are uploaded
    if not video1 or not video2:
        raise BadRequest('Please upload both videos.')

    # Calculate the similarity percentage
    similarity_percentage = calculate_similarity(video1_path, video2_path)

    # Pass the video filenames and similarity percentage to the template
    return render_template('compare.html', video1=video1, video2=video2, similarity_percentage=similarity_percentage)


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.run(debug=False)
