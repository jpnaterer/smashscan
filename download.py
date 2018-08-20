from datetime import datetime
import pytube
import os
import argparse

# https://python-pytube.readthedocs.io/en/latest/api.html
def download_video(video_id, save_name, save_dir, tmp_dir):
    start_time = datetime.now()
    print("Downloading video with video_id=%s" % video_id)

    yt = pytube.YouTube("https://www.youtube.com/watch?v=" + video_id)
    file_path = "%s/%s" % (save_dir, save_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

    stream, stream_resolution = None, None
    video_resolutions = ["360p", "240p", "480p", "720p", "1080p", "144p"]
    for video_resolution in video_resolutions:
        stream = yt.streams.filter(
            file_extension="mp4", resolution=video_resolution).first()
        if stream:
            stream_resolution = video_resolution
            break

    stream_title = stream.default_filename
    print("Successfully obtained stream with:\n\tresolution=%s"
        "\n\tsize=%.2fMB\n\tname=%s" % (stream_resolution,
        stream.filesize/1024/1024, stream_title))
    stream.download(tmp_dir)
    os.rename(tmp_dir + stream_title, file_path)

    finish_time = (datetime.now() - start_time).total_seconds()
    print("Successfully downloaded video in %.2fs" % finish_time)
    return file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download YouTube videos \
        using the PyTube library. The (360p, mp4) is the default format.')
    parser.add_argument('video_id', type=str, 
        help='The YouTube video id to download.')
    parser.add_argument('save_name', type=str, 
        help='The file name to be saved.')
    parser.add_argument('save_dir', type=str, default='videos', nargs='?',
        help='The file directory to be used.')
    parser.add_argument('tmp_dir', type=str, default='/tmp/', nargs='?',
        help='The directory to be used for temporarily saving the video.')
    args = parser.parse_args()
    download_video(args.video_id, args.save_name, args.save_dir, args.tmp_dir)
