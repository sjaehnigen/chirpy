#!/bin/bash

check(){
    if [ -z `echo $1 | egrep "test_test|tearDown|setUp|_sync_class"` ]; then
        _a=`cat ../test/*.py | grep $1`
        [ -z "$_a" ] && echo "No test found for ${1:5}"
    fi
}
for _m in `find .. -name "*py" | xargs grep "def " | egrep -v 'bin/|external/|__|test.py' | awk -F '(' '{print $1}' | awk '{print "test_"$NF}' | sort`; do 
	check $_m
done
