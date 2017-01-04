#!/bin/sh

echo "Learning squares:"
python src/learn.py square.png
echo "Detecting squares:"
python src/detect.py square.png.rtable detect.png detect.square.acc.png
echo "Accumulator image saved in detect.square.acc.png."

echo "Learning circles:"
python src/learn.py circle.png
echo "Detecting circles:"
python src/detect.py circle.png.rtable detect.png detect.circle.acc.png
echo "Accumulator image saved in detect.circle.acc.png."
