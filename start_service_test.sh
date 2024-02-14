#!/bin/sh


streamlit run stEnt_testing.py &

# Keep the script running to keep the container alive
tail -f /dev/null
