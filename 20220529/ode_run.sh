#! /bin/bash
rm -rf ODElimit1000T20
python -m grid ODElimit1000T20 "python ODE.py --d 2 --p 1 --T 50 --N 1 --a 1 --Ni 0.83972 -1.40140" --dt 0.0001
