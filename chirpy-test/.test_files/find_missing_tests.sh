#!/bin/bash

check(){
    if [ -z `echo $1 | egrep "align_atoms|wrap_molecules|join_molecules|import_submodules|vector|triple_product|tracked_update|signal_filter|rdf|PDF|Iterator|test_test|tearDown|setUp|_sync_class|main|write|print|Parse|WhatIsThis|symbols_to|spline|set_keyword|read_|isHB|load_template|magnetic_dipole_shift_origin|Projection|from_data|electric_dipole_shift_origin|arr_head_sense|cumulate_hydrogen_bonding_events|set_name|get_v|get_p_and_v|get_positions|g09_|extract_mtm_data_tmp|ExtractFrame|E_nm2J|E_J2nm|E_Hz2nm|draw_"` ]; then
        _a=`cat ../*.py | grep $1`
        [ -z "$_a" ] && echo "No test found for ${1:5}"
    fi
}
# for _m in `find ../.. -name "*py" | egrep -v 'interface|visualise|classes|create|bin' | xargs grep "def " | egrep -v ' _|#|__|test.py' | awk -F '(' '{print $1}' | awk '{print "test_"$NF}' | sort`; do 
for _m in `find ../.. -name "*py" | egrep -v 'visualise|classes' | xargs grep "def " | egrep -v ' _|#|__|test.py' | awk -F '(' '{print $1}' | awk '{print "test_"$NF}' | sort`; do 
	check $_m
done
