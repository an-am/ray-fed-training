#!/bin/bash
# run_all.sh
# This script opens a new Terminal window for each party (keys 0 through 9)

for i in {0..9}; do
  osascript -e "tell application \"Terminal\" to do script \"python /Users/antonelloamore/PycharmProjects/ray-fed-training/test.py $i\""
done