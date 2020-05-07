"""Trains many models at once through subprocess for isolation
"""
import subprocess

for i in [1,2,5,10,15,20,30,50,70,80,100]:
	subprocess.run(["python3.7 train.py %d"%i], shell=True)