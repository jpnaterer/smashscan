# SmashScan

This project is intended to perform video analysis on pre-recorded footage of the game Super Smash Bros Melee. Once the project has progressed enough, I plan to integrate it with the [SmashVods](http://smashvods.com/) video database.

## TODO List
If you are interested in helping out, join this repository's [Slack](https://join.slack.com/t/smashscan/shared_invite/enQtNDE5MjA5OTI0NDgwLTYwNGNkOWFmZjRjYjkwNDRkNzMzZGJjZjQwZTY5Y2YwZDhmNDJiYzEyZjk1OWJmMmU2YzYzNjRjMTIzYmM2YTI) and send a message.
+ Create a final bounding box output, that averages the boxes to create an output box and removes outliers
  + Using this bounding box, output the times that each stage is present within a video. Try to ignore highlight clips.
+ Research ways to detect stock icons (Either template matching, feature matching, or neural networks)
+ Research ways to detect the timer (Either template matching, feature matching, or neural networks)
+ Research ways to detect the percent counter (Either template matching, feature matching, or neural networks)
+ Add additional game capabilities (SSB64, SSB4, PM, Rivals)

### Milestones
1. [Find legal stages regardless of overlay.](https://medium.com/@seft/smashscan-using-neural-networks-to-analyze-super-smash-bros-melee-a7d0ab5c0755)

## Instalation Guide
1. Install TensorFlow-GPU. I followed [Mark Jay's tutorial](https://www.youtube.com/watch?v=vxjbL5iN1XY) for Ubuntu 18.04. For reference my NVIDIA driver is 390.77, my CUDA version is 9.0.176, CUDANN version is 7.0.5, and tensorflow-gpu version is 1.5. I had the most trouble installing this and getting test tensorflow-gpu examples to work. Make sure to install these in the correct order, and I'd avoid installing the newest versions of these drivers/packages. I tried and couldn't get the test examples to work.
2. Install the [DarkFlow](https://github.com/thtrieu/darkflow) repository globally. I followed [Mark Jay's tutorials](https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM) to understand the basics of the DarkFlow. 
3. Clone this repo and create an `annotations`, `output`, and `videos` folder inside. The `.gitignore` doesn't like committing "empty" folders.

## Testing Guide

Obtain the YouTube video-id for the video you wish to test on. The video-id is the set of characters after the "v=" in a YouTube video URL. Afterwards, use `download.py` to save the video to the `videos` directory.

``` bash
# https://www.youtube.com/watch?v=dQw4w9WgXcQ
python download.py dQw4w9WgXcQ example-video-name.mp4
```

Use `test.py` to draw bounding boxes on the video. This uses the pre-trained weights in the `cfg` folder.

``` bash
python test.py example-video-name.mp4

# To output example frames (.png's), add the save flag.
python test.py example-video-name.mp4 -save

# If you have imagemagick installed, create a .gif with the following command.
cd output
convert -delay 5 -loop 0 *.png Darkflow_SSBM_v1.gif
```


## Training Guide

Obtain the YouTube video-id for the video you wish to train on. The video-id is the set of characters after the "v=" in a YouTube video URL. Afterwards, use `download.py` to save the video to the `videos` directory.

``` bash
# https://www.youtube.com/watch?v=dQw4w9WgXcQ
python download.py dQw4w9WgXcQ example-video-name.mp4
```

Use `label.py` to save frames to the `images` directory and labels to the `annotations` directory. The key-bindings used for `label.py` are listed below.

``` bash
python label.py example-video-name.mp4
```

`left` - move backward 1 frame  
`right` - move forward 1 frame  
`alt+direction` - move 10 frames  
`ctrl+direction` - move 100 frames  
`shift+direction` - move 1000 frames  

`left-click -> release` - draw bounding box  
`r` - begin recording once you are satisfied with the box drawn  
`enter` - end recording and specify stage number in terminal  
`` ` `` - exit the labeler and save annotations & frames  
`q` - exit the labeler without saving annotations & frames  

Opening the `images` folder freezes my computer because of how many images it tries to display at once. If your running Linux, a useful command to run to see the most recently updated file in a directory is as follows.

```bash
ls -t images | head -1

# Eye of GNOME graphics viewer. Just an image viewer program.
eog 123456.png
```

Once you are satisfied with the new labels, move the `images` and `annotations` folders into the `train` folder of your `darkflow` installation directory. Then run the following two commands in separate terminals.

```bash
# Training command.
flow --model cfg/tiny-yolo-voc-6c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/annotations --dataset train/images --epoch 100 --gpu 1.0 --summary summary

# Optional command to track the progress visually.
tensorboard --logdir summarytrain
```

After some time, you can cancel training by hitting `ctrl+c`. The DarkFlow trainer saves checkpoints every couple minutes, so you aren't losing much progress. If I plan on training for a long period of time (over 12 hours), I'll typically set the checkpoints to be saved over a longer time span. AKA, I will add the flag `--save 40000`. which will save checkpoints every 2500 iterations (40000/16, where batch size = 16).

This is because DarkFlow has an issue with gradient descent over long periods of time and training will get stuck at NaN. DarkFlow's `default.py` recommends rmsprop, but I tried adam and sgd with no success. It seems to be a common issue with the trainer from the issues I've read on the repository :/

Next, convert the most recent checkpoint into something usable by SmashScan.

``` bash
flow --model cfg/tiny-yolo-voc-6c.cfg --load -1 --savepb
```

The .pb (Protocol Buffer) format is a method for serializing structured data. The DarkFlow library saves the .meta and .pb files to the `built_graph` directory. Copy both of these files into the `cfg` directory of the SmashScan directory.

### Label History
Below are a list of tournaments I trained on for the initial release. The size of these annotation-image pairs is >10GB, so I don't plan on uploading them anytime soon. This list is mainly for referencing what tournaments have been used.

| Tournament | Annotation Range | Frames |
| :--------: | :----------: | :--------------: |
| GOML 2018 | 1 - 5019 | 5019 |
| EVO 2018 | 5020 - 8274 | 3255 |
| Low Tier City 6 | 8275 - 9246 | 972 |
| Smash Factor 7 | 9247 - 12485 | 3239 |
| Saints Gaming Live 2018 | 12486 - 14019 | 1534 |
| Smash N Splash 4 | 14020 - 15720 | 1701 |
| Dreamhack Austin 2018 | 15721 - 17299 | 1579 |
| Combo Breaker 2018 | 17300 - 19091 | 1792 |
| Momocon 2018 | 19092 - 20562 | 1471 |
| Pound Underground 2018 | 20563 - 22747 | 2185 |
| Smash Summit 6 | 22747 - 24852 | 2106 |
| Aegis 2018 | 24853 - 26005 | 1153 |
| Flatiron 3 | 26006 - 27383 | 1378 |
| Noods Oakland | 27384 - 29554 | 2171 |
| The Mango | 29555 - 30976 | 1422 |
| No Fun Allowed 2 | 30977 - 33133 | 2157 |
| The Gang | 33134 - 34861 | 1730 |
| Noods Noods Noods | 34862 - 36760 | 1899 |
| Valhalla | 36761 - 38503 | 1743 |

