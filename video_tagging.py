import argparse
from math import floor


def plot_predict(pred_array, name, frame_skip, tag_names_video, save_path):
    print(pred_array.shape)
    plt.rcParams['figure.figsize'] = (15, 6)
    fig, ax = plt.subplots(1)

    x_lims = list(map(dt.datetime.utcfromtimestamp, [0, np.shape(pred_array)[1]]))
    x_lims = dates.date2num(x_lims)
    y_lims = [np.shape(pred_array)[0] - 1, 0]

    im = ax.imshow(pred_array, extent=[x_lims[0], x_lims[1], y_lims[0] + 0.5, y_lims[1] - 0.5], interpolation='none',
                   aspect='auto', origin='lower', cmap='magma')
    im.set_clim(0, 1)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('time (mm:ss), frame skip: ' + str(frame_skip), fontsize=12)
    ax.set_yticks(np.arange(len(tag_names_video)))
    ax.set_yticklabels(tag_names_video, fontsize=9)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    cbar = fig.colorbar(im, ticks=[0., 0.5, 1.0])
    cbar.set_label('Probability', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)


def video_tagging(args):
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    try:
        with open(args.classes_path, 'r') as f:
            tag_names_video = [line.strip() for line in f]
    except IOError:
        print("File with classes not exists!")
        return

    if not os.path.isfile(args.model_path):
        print("File with model not exists!")
        return

    video_path = args.video_path
    if video_path[-1] == "\\" or video_path[-1] == "/":
        video_path = video_path[0:len(video_path)-1]

    if not os.path.isdir(video_path):
        if not os.path.isfile(video_path):
            print("Video file not exists!")
            return



    predict_video(args, args.video_path, tag_names_video)


def add_audio(args, video_writer_name, video_with_audio_name, final_video_name):
    video_without_audio = mpe.VideoFileClip(video_writer_name)
    video_with_audio = mpe.VideoFileClip(video_with_audio_name)
    audio = video_with_audio.audio
    final_video = video_without_audio.set_audio(audio)

    if args.video_bitrate == str(0):
        video_height = video_with_audio.size[1]
        video_bit_rate = video_height * 6.5
        video_bit_rate = video_bit_rate * 1000
    else:
        video_bit_rate = args.video_bitrate

    if args.gpu_encode:
        final_video.write_videofile(final_video_name, audio_codec='libvorbis', codec='mpeg4',
                                    bitrate=str(video_bit_rate),
                                    ffmpeg_params=['-vcodec', 'h264_nvenc', '-preset', 'slow'], preset='medium')
    else:
        final_video.write_videofile(final_video_name, audio_codec='libvorbis', codec='mpeg4',
                                    bitrate=str(video_bit_rate), preset='medium')

    video_without_audio.close()
    video_with_audio.close()


def predict_video(args, predict_path, tag_names_video, image_size=(299, 299)):

    model_video = tensorflow.keras.models.load_model(args.model_path)

    output_files_dir = arguments.output_path

    if os.path.isdir(predict_path):
        files = os.listdir(predict_path)
    else:
        files = [os.path.basename(predict_path)]
        predict_path = os.path.dirname(predict_path) + "/"

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    predList = {}

    for file in files:

        
        prev_status = -1
        filename_woext = os.path.splitext(file)[0]
        file_name = ''.join([predict_path, file])
        print("Processing video:")
        print(file_name)
        if not os.path.isdir(file_name):  
            if not filename_woext in arguments.skip_videos:
                cap = cv2.VideoCapture(file_name)

                frame_skip = int(cap.get(cv2.CAP_PROP_FPS))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
                fps = (cap.get(cv2.CAP_PROP_FPS))

                duration = floor(n_frames / fps)

                font = cv2.FONT_HERSHEY_SIMPLEX
                longest_name = max(tag_names_video, key=len)
                text_to_print = longest_name + ": " + "{:.1f}".format(100) + "%"
                font_scale = 1
                font_thickness = 1
                text_size, _ = cv2.getTextSize(text=text_to_print, fontFace=font, fontScale=font_scale,
                                            thickness=font_thickness)

                font_scale = (width / 2) / text_size[0]
                if font_scale > 1.0:
                    font_thickness = int(round(font_scale))

                text_size, _ = cv2.getTextSize(text=text_to_print, fontFace=font, fontScale=font_scale,
                                            thickness=font_thickness)

                if args.show_images:
                    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                    window_height = int(round((600 / width) * height))
                    cv2.resizeWindow('Video', 600, window_height)

                pred_list = []
                annotations_dict = {}
                sample_factor = fps / frame_skip
                actual_n_frame = 0

                if args.save_video:
                    if not os.path.exists(output_files_dir):
                        os.makedirs(output_files_dir)
                    output_file_dir = output_files_dir + filename_woext + "/"
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)
                    video_writer_name = output_file_dir + filename_woext + "NoSound.mp4"
                    final_video_name = output_file_dir + filename_woext + ".mp4"
                    video_write = cv2.VideoWriter(video_writer_name, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
                seconds = 0
                # while actual_n_frame < n_frames:
                for f in tqdm(range(n_frames), unit=' frames', file=sys.stdout):
                    _, frame = cap.read()
                    if not _:
                        break
                    actual_msecs = cap.get(cv2.CAP_PROP_POS_MSEC)

                    if (cap.get(cv2.CAP_PROP_POS_MSEC)/1000) >= seconds:
                        seconds += 1
                        output = cv2.resize(frame, image_size)
                        output = (output[..., ::-1].astype(np.float32)) / 255.0
                        output = np.expand_dims(output, axis=0)
                        pred = np.squeeze(model_video.predict(output))
                        predList = {}
                        for index in range(0, len(pred)):
                            predList[index] = pred[index]
                        predictions_sorted = sorted(predList, key=predList.get, reverse=True)
                        predictions_sorted_top = predictions_sorted
                        pred_list.append(pred)
                        for index, prediction in enumerate(pred):
                            if tag_names_video[index] in annotations_dict:
                                annotations_dict[tag_names_video[index]].append(
                                    {'time': np.float64(actual_msecs), 'prediction': np.float64(prediction)})
                            else:
                                annotations_dict[tag_names_video[index]] = []
                                annotations_dict[tag_names_video[index]].append(
                                    {'time': np.float64(actual_msecs), 'prediction': np.float64(prediction)})

                    if not os.path.exists(output_files_dir):
                        os.makedirs(output_files_dir)
                    output_file_dir = output_files_dir + filename_woext + "/"
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)

                    texts_to_print = {}
                    if predList[predictions_sorted_top[0]] > 0.0:
                        for index, predictionIndex in enumerate(predictions_sorted_top):
                            if predList[predictions_sorted_top[index]] > 0.1:
                                texts_to_print[tag_names_video[predictionIndex]] = predList[predictionIndex] * 100
                    if texts_to_print: 
                        longest_name = max(texts_to_print, key=len)
                        text_to_print = longest_name + ": " + "{:.1f}".format(100) + "%"

                        text_size, _ = cv2.getTextSize(text=text_to_print, fontFace=font, fontScale=font_scale, thickness=font_thickness)
                        num_of_texts = len(texts_to_print)

                        text_frame_width = text_size[0] + 10
                        text_frame_height = int(round(num_of_texts * (text_size[1] + (5 + 16 + font_thickness) * font_scale)))

                        rectangle_frame = frame.copy()
                        cv2.rectangle(rectangle_frame, (0, 0), (text_frame_width, text_frame_height), (0, 0, 0), -1)
                        alpha = 0.4
                        frame = cv2.addWeighted(rectangle_frame, alpha, frame, 1 - alpha, 0)

                        num_of_texts = 0
                        for text_to_print in texts_to_print:
                            num_of_texts = num_of_texts + 1
                            actual_probability = texts_to_print[text_to_print]
                            text_to_print = text_to_print + ": " + "{:.1f}".format(actual_probability) + "%"
                            cv2.putText(frame, text_to_print,
                                        (5, int(round(num_of_texts * (text_size[1] + (16 + font_thickness) * font_scale)))),
                                        font, font_scale, (66, 185, 245), font_thickness, cv2.LINE_AA)

                    if args.save_video:
                        video_write.write(frame)

                    if args.show_images:
                        cv2.imshow("Video", frame)
                        cv2.waitKey(1)

                    # actual_status = (actual_n_frame / n_frames) * 100
                    # if prev_status + 1 < actual_status:
                    #     print(str(int(round(actual_status))) + " %")
                    #     prev_status = actual_status

                    actual_n_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                if len(pred_list) > 0:
                    pred_array = np.rot90(pred_list)
                    if sample_factor > 1:
                        sample_factor = int(round(sample_factor))
                        pred_array = pred_array[:, ::sample_factor]
                    else:
                        sample_factor = round(1 / sample_factor)
                        pred_array = np.repeat(pred_array, sample_factor, axis=1)

                    plot_save_path = output_file_dir + filename_woext + '.png'
                    if args.plot_predict:
                        plot_predict(pred_array, filename_woext, frame_skip, tag_names_video, plot_save_path)

                if args.save_video:
                    video_write.release()
                    add_audio(args, video_writer_name, file_name, final_video_name)
               
                keys = []
                for key in annotations_dict:
                    keys.append(key)
                if len(annotations_dict[keys[0]]) == duration:
                    pass
                else:
                    for key in keys:
                        annotations_dict[key] = annotations_dict[key][:duration]

                with open(output_file_dir + filename_woext + '.json', 'w') as outfile:
                    json.dump(annotations_dict, outfile)

                # if os.path.exists(video_writer_name):
                #     os.remove(video_writer_name)

    print("All videos done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Tagging')
    parser.add_argument('--video_path', type=str, default="C:/path/to/videos/",
                        help='path to video or folder with videos for tagging')
    parser.add_argument('--model_path', type=str, default="vasulka-video.h5",
                        help='path to trained video model')
    parser.add_argument('--classes_path', type=str, default="classes.txt",
                        help='path to txt file with classes')
    parser.add_argument('--output_path', type=str, default="output/",
                        help='path for output folder')
    parser.add_argument('--cpu', type=bool, default=False,
                        help='if true, tagging runs on CPU')
    parser.add_argument('--gpu_encode', type=bool, default=False,
                        help='if True, video encode runs on GPU') 
    parser.add_argument('--video_bitrate', type=str, default=str(0),
                        help='final video bitrate, if 0 bit rate is set to source video height * 4.5')
    parser.add_argument('--show_images', type=bool, default=False,
                        help='if true, processed frames will be displayed')
    parser.add_argument('--save_video', type=bool, default=False,
                        help='if true, processed video will be saved with printed annotation in video')
    parser.add_argument('--plot_predict', type=bool, default=False,
                        help='if true, preddictions will be write to plot and saved to PNG image')
    parser.add_argument('--skip_videos', action='store',
                        type=str, nargs='*', default=[],
                        help="examples: --skip_videos video1.mp4 video2.mp4, --skip_videos video3.mp4")
    arguments, _ = parser.parse_known_args()

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf    
    import cv2
    import numpy as np
    from tqdm import tqdm
    import tensorflow.keras
    import datetime as dt
    from matplotlib import pyplot as plt
    from matplotlib import dates
    import json
    import moviepy.editor as mpe    
    import sys

    video_tagging(arguments)