#!/bin/bash

for _m in `find $1 -name "*py" | xargs grep "def " | egrep -v 'bin/|external/|__|test.py' | awk -F '(' '{print $1}' | awk '{print "test_"$NF}' | sort`; do 
	[ ! -z "`grep $_m $1/test.py`" ] || echo "No test found for ${_m:5}"
done
