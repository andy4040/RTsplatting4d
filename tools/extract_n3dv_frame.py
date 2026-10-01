"""Extract one decoded frame per N3DV camera without resizing or modifying videos.

Example (frame indices start at zero):
    python tools/extract_n3dv_frame.py --input /datasets/coffee_martini.zip \
        --output data/coffee_martini_f000/images --frame 0

Input may be a ZIP or a directory containing camNN.mp4 files (nested folders
are supported). Equal frame indices assume the input cameras are synchronized;
this script does not estimate or correct camera timing, poses, or geometry.
PNG stores the decoded video pixels losslessly, at their original resolution.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile


CAMERA = re.compile(r"cam\d+\.mp4", re.IGNORECASE)


def decode_frame(video, index):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        # Decode sequentially: frame seeking can depend on codec/keyframes.
        for position in range(index + 1):
            ok, pixels = capture.read()
            if not ok or pixels is None:
                raise ValueError(f"Video has no decodable frame {index}: {video} (stopped at {position})")
        return pixels
    finally:
        capture.release()


def extract(source, output, frame=0):
    import cv2

    source, output = Path(source).resolve(), Path(output).resolve()
    if frame < 0:
        raise ValueError("Frame index must be nonnegative (zero-based)")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if source.is_dir():
        context = nullcontext(None)
    elif source.is_file() and zipfile.is_zipfile(source):
        context = zipfile.ZipFile(source)
    else:
        raise ValueError("Input must be a directory or ZIP containing camNN.mp4 files")

    with context as archive:
        if archive is None:
            videos = [(path.stem.lower(), path) for path in source.rglob('*')
                      if path.is_file() and CAMERA.fullmatch(path.name)]
        else:
            videos = [(PurePosixPath(item.filename).stem.lower(), item)
                      for item in archive.infolist() if not item.is_dir()
                      and CAMERA.fullmatch(PurePosixPath(item.filename).name)]
        videos.sort(key=lambda item: (int(item[0][3:]), item[0]))
        if not videos:
            raise ValueError("No camNN.mp4 files found")
        ids = [int(name[3:]) for name, _ in videos]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate camera IDs; select a single scene as input")

        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after all cameras succeed. Temporary data stay next to
        # the requested output and are removed if decoding fails.
        with tempfile.TemporaryDirectory(prefix='.n3dv_extract_', dir=output.parent) as temporary:
            temporary = Path(temporary)
            staged = temporary / 'images'
            staged.mkdir()
            for name, entry in videos:
                if archive is None:
                    video = entry
                else:
                    # Stream just one video; never unpack paths supplied by ZIP.
                    video = temporary / 'camera.mp4'
                    with archive.open(entry) as reader, video.open('wb') as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                pixels = decode_frame(video, frame)
                ok, encoded = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not ok:
                    raise RuntimeError(f"PNG encoding failed: {name}")
                (staged / f'{name}.png').write_bytes(encoded.tobytes())
                height, width = pixels.shape[:2]
                print(f'{name}: frame {frame}, {width}x{height}', flush=True)
                if archive is not None:
                    video.unlink()
            if output.exists():
                raise FileExistsError(f"Output appeared during extraction: {output}")
            staged.rename(output)
    print(f'Saved {len(videos)} PNG images to {output}', flush=True)
    return len(videos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='Single-scene ZIP or video directory')
    parser.add_argument('--output', required=True, type=Path, help='New directory for camNN.png files')
    parser.add_argument('--frame', type=int, default=0, help='Zero-based decoded frame index (default: 0)')
    args = parser.parse_args()
    try:
        extract(args.input, args.output, args.frame)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
