from __future__ import unicode_literals
import os
import argparse
import youtube_dl

# https://github.com/rg3/youtube-dl
# View video formats with bash: youtube-dl -F "video_id"
def download_video(video_id, save_name, save_dir):

    output_file_path = '%s/%s' % (save_dir, save_name)
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height=360]\
            [protocol!=http_dash_segments]/best[ext=mp4][height=360]\
            [protocol!=http_dash_segments]',
        'outtmpl': output_file_path
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_id])
    return output_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download YouTube videos \
        using the youtube-dl library. The (360p, mp4) is the default format.')
    parser.add_argument('video_id', type=str, 
        help='The YouTube video id to download.')
    parser.add_argument('save_name', type=str, 
        help='The file name to be saved.')
    parser.add_argument('save_dir', type=str, default='videos', nargs='?',
        help='The file directory to be used.')
    args = parser.parse_args()
    download_video(args.video_id, args.save_name, args.save_dir)
