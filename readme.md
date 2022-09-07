# mediaArtLiveArchive – Video Tagging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a console application used for video content tagging in videos.

<img src="./img/violinPower.png" alt="TACR" width="100%" max-width="720px"/>

## Running the console application
Example 1 – Tagging just one video
```
python video_tagging.py --video_path "path/to/video/video.mp4"
```
Example 2 – Tagging multiple videos in one folder
```
python video_tagging.py --video_path "path/to/videos"
```

## video_tagging.py arguments
|argument|description|type|default|
|---|---|---|---|
|`--video_path`|path to video or folder with videos for tagging|str|"C:/path/to/videos/"|
|`--model_path`|path to trained video model|str|"vasulka-video.h5"|
|`--classes_path`|path to txt file with classes|str|"classes.txt"|
|`--output_path`|output path for predicitions|str|"output/"|
|`--cpu`|if True, tagging runs on CPU|bool|False|
|`--gpu_encode`|if True, video encoding runs on GPU|bool|False|
|`--video_bitrate`|final video bitrate, if 0 bit rate is set to source video height * 4500|str|0|
|`--show_images`|if true, processed frames will be displayed|bool|False|
|`--save_video`|if True, annotations are rendered to video|bool|False|
|`--plot_predict`|if True, prediction plot is saved to image|bool|False|
|`--skip_videos`|skip selected videos in video_path|str|[]|

## The pre trained model on Vasulka's database can be downloaded here:
* Model 1 (Xception): [video_Vasulka.zip](https://vasulkalivearchive.net/models/video_Vasulka.zip) 
* Model 2 (NasNetLarge): [video_Vasulka2.zip](https://vasulkalivearchive.net/models/video_Vasulka2.zip) 
### The model allows you to tag these categories (metric f1-score [-]):


|                          | Model 1 |  | Model 2 |
| ------------------------ | ------- |  | ------- |
| Body                     | 0.865   |  | 0.855   |
| Digit                    | 0.909   |  | 0.889   |
| Effect                   | 0.505   |  | 0.305   |
| Air                      | 0.712   |  | 0.758   |
| Earth                    | 0.661   |  | 0.632   |
| Fire                     | 0.704   |  | 0.437   |
| Water                    | 0.676   |  | 0.714   |
| Face                     | 0.910   |  | 0.897   |
| Keying                   | 0.593   |  | 0.627   |
| Interior                 | 0.819   |  | 0.777   |
| Landscape                | 0.769   |  | 0.743   |
| Letter                   | 0.923   |  | 0.899   |
| Car                      | 0.920   |  | 0.920   |
| TV set                   | 0.758   |  | 0.640   |
| Machine vision (fisheye) | 0.894   |  | 0.859   |
| Rutt/Etra Scan processor | 0.571   |  | 0.769   |
| Steina                   | 0.340   |  | 0.413   |
| Stripes                  | 0.624   |  | 0.718   |
| Violin                   | 0.535   |  | 0.502   |
| Woody                    | 0.942   |  | 0.923   |

## Dependencies
```
tensorflow==2.3.1
moviepy==1.0.3
opencv_python==4.3.0.36
matplotlib==3.2.2
numpy==1.18.5
```

### Reference
This repository uses two models. First is [Xception] model proposed in:  
Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (p. 1251-1258).

Second is [NasNetLarge] model proposed in:  
ZOPH, Barret, et al. (2018). Learning transferable architectures for scalable image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. p. (8697-8710).

### Cite
Sikora, P. (2022). *MediaArtLiveArchive – Video Tagging* [Software]. https://github.com/vasulkalivearchive/video

### Acknowledgements
The MediaArtLiveArchive – Video Tagging software was implemented with the financial participation of the Technical Agency of the Czech Republic under the ÉTA programme. It is an output of the project Media Art Live Archive: Intelligent Interface for Interactive Mediation of Cultural Heritage (No. TL02000270). 

<!-- [![plot](./img/logo_TACR_zakl.png)](https://www.tacr.cz/) -->
<a href="https://www.tacr.cz/">
    <img src="./img/logo_TACR_zakl.png" alt="TACR" width="200"/>
</a>