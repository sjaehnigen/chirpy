#!/bin/bash

check(){
    if [ -z `echo $1 | egrep "align_atoms|wrap_molecules|join_molecules|import_submodules|vector|triple_product|tracked_update|signal_filter|rdf|PDF|Iterator|test_test|tearDown|setUp|_sync_class"` ]; then
        _a=`cat ../*.py | grep $1`
        [ -z "$_a" ] && echo "No test found for ${1:5}"
    fi
}
for _m in `find ../.. -name "*py" | egrep -v 'interface|visualise|external|classes|create|bin' | xargs grep "def " | egrep -v ' _|#|__|test.py' | awk -F '(' '{print $1}' | awk '{print "test_"$NF}' | sort`; do 
	check $_m
done
