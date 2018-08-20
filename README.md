# SmashScan

This project is intended to perform video analysis on pre-recorded footage of the game Super Smash Bros Melee. Once the project has progressed enough, I plan to integrate it with the [SmashVods](http://smashvods.com/) video database.

## TODO List
If you are interested in helping out, join this repo's [Slack](https://join.slack.com/t/smashscan/shared_invite/enQtNDE5MjA5OTI0NDgwLTYwNGNkOWFmZjRjYjkwNDRkNzMzZGJjZjQwZTY5Y2YwZDhmNDJiYzEyZjk1OWJmMmU2YzYzNjRjMTIzYmM2YTI) and send a message.
+ Create a final bounding box output, that averages the boxes to create an output box and removes outliers
  + Using this bounding box, output the times that each stage is present within a video. Try to ignore highlight clips.
+ Research ways to detect stock icons (Either template matching, feature matching, or neural networks)
+ Research ways to detect the timer (Either template matching, feature matching, or neural networks)
+ Research ways to detect the percent counter (Either template matching, feature matching, or neural networks)
+ Add additional game capabilities (SSB64, SSB4, PM, Rivals)

## Training Guide

Obtain the YouTube video-id for the video you wish to train on. The video-id is the set of characters after the "v=" in a YouTube video URL. Afterwards, use `download.py` to save the video to the `videos` directory.

``` bash
# https://www.youtube.com/watch?v=dQw4w9WgXcQ
python download.py dQw4w9WgXcQ example-video-name.mp4
```

Use `label.py` to save frames to the `images` directory and labels to the `annotations` directory. The keybindings used for `label.py` are listed below.

``` bash
python label.py example-video-name.mp4
```

>`left` - move backward 1 frame  
`right` - move forward 1 frame  
`alt+direction` - move 10 frames  
`ctrl+direction` - move 100 frames  
`shift+direction` - move 1000 frames  

>`left-click -> release` - draw bounding box  
`r` - begin recording once you are satisfied with the box drawn  
`enter` - end recording and specify stage number in terminal  
`` ` `` - exit the labeler and save annotations & frames  
`q` - exit the labeler without saving annotations & frames  

Opening the `images` folder freezes my computer because of how many images it tries to display at once. If your running linux, a useful command to run to see the most recently updated file in a directory is as follows.

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

After some time, you can cancel training by hitting `ctrl+c`. The DarkFlow trainer saves checkpoints every couple minutes, so you aren't losing much progress. Next, convert the most recent checkpoint into something usable by SmashScan.

``` bash
flow --model cfg/tiny-yolo-voc-6c.cfg --load -1 --savepb
```

The .pb (Protocol Buffer) format is a method for serializing structured data. The DarkFlow library saves the .meta and .pb files to the `built_graph` directory. Copy both of these files into the `cfg` directory of the SmashScan directory.

### Label History

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

