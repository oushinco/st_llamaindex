#!/bin/sh


streamlit run stEnt.py &

# Keep the script running to keep the container alive
tail -f /dev/null
