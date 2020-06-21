import sys
from flask import Flask,request
from flask_cors import CORS
import json
import base64
import ffmpeg
import subprocess
import os
import errno
import time
from predict import get_prediction
sys.path.append("./i3d/experiments/ucf101")
#import test_flow_result
import test_rgb_flow_result

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/automl", methods=['POST'])
def callAutoml():
    print "automl"
    print(request.json)
    data=request.json
    print(data["result"])
    project_id="extreme-tribute-220411"
    model_id="TRL2058776748956521335"
    ans=get_prediction(data["result"], project_id, model_id)
    ans="{"+ans+"}"
    #print(ans)
    return json.dumps(ans)

@app.route("/api/v1/predict_sign", methods=['POST'])
def predict_sign_language():
    print "mp4file"
    # recieve data and split it
    data=request.json
    file_data=data["file"].split(",")
    file_name="./videos_no_process/%s"%(data["filename"])
    # decode base64 mp4 file and save it to server
    file_data_content=base64.b64decode(file_data[1])
    save_file=open(file_name,"wb")
    save_file.write(file_data_content)
    save_file.close()
    # use ffmpeg to transfer mp4 file
    pfile_name="./videos/test%s"%(data["filename"])
    process_video_file(file_name,data["filename"])
    # create file path
    dir,dir_x,dir_y,dir_im=create_dir_for_videos(data["filename"])
    # do dense-flow
    cmd="sudo ./dense-flow/build/denseFlow_gpu --vidFile='%s' --xFlowFile='%s/flow_x' --yFlowFile='%s/flow_y' --imgFile='%s/im' --bound=16 --type=2 --device_id=0 --step=2"%(pfile_name,dir_x,dir_y,dir_im)
    print(cmd)
    proc=subprocess.Popen(cmd,shell=True)
    proc.wait()
    print(dir)
    # translate
    #ans=test_flow_result.run_training(dir)
    ans=test_rgb_flow_result.run_training(dir)
    return ans

def process_video_file(file_name,f):
    stream = ffmpeg.input(file_name)
    stream = ffmpeg.filter(stream, 'fps', fps=25)
    stream = ffmpeg.output(stream, "./videos/test%s"%(f))
    ffmpeg.run(stream,overwrite_output=True)

def create_dir_for_videos(file_name):
    dir_name=file_name.split(".")
    dir="./videos_to_images/test%s"%(dir_name[0])
    dir_x="./videos_to_images/test%s/x"%(dir_name[0])
    dir_y="./videos_to_images/test%s/y"%(dir_name[0])
    dir_im="./videos_to_images/test%s/i"%(dir_name[0])
    try:
        os.mkdir(dir)
        os.mkdir(dir_x)
        os.mkdir(dir_y)
        os.mkdir(dir_im)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
            pass
        else:
            raise
    return dir,dir_x,dir_y,dir_im

if __name__ == "__main__":
    #test_flow_result.load_model()
    #test_flow_result.run_training("/var/www/html/web/videos/test1")
    test_rgb_flow_result.load_model()
    app.run(host='0.0.0.0', port=1234)
