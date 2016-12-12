#!/bin/bash

#Job variables
#Job variables
ROOT_DIR=/scratch/aolpin/project
MEMORY=1G
CPU=2
WALL_TIME=15m
datafile=/scratch/aolpin/project/preprocessed-json.txt
#statics
test_name='test1'
out_directory="${ROOT_DIR}/${test_name}"
#Variables
c=( 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 1 2 3 4 5 6 7 8 9 10 )
loss=( 'hinge' 'squared_hinge' )
multi=( 'ovr' 'crammer_singer' )
tol=( 0.00001 0.0001 0.001 0.01 0.1 )
iterations=( 1000 2000 3000 4000 5000 6000)
inter=( 0.001 0.01 0.1 0.5 1 2 3 4 5 6 7 8 9 10 )


function check_directory(){
	if [ -z "$1" ]                           
	then
	     echo "No directory variable passed in"  
	else
	     if [ ! -d $1 ]
		then
		    echo "Creating directory: $1"
		    mkdir -p $1
		   else
		   	echo "Directory: $1 exists"
		fi
	fi
}


#Runs elastic regularization
function run_tests(){
	iter=1
	check_directory "${out_directory}"
	for e in "${iterations[@]}"; do
	    for l in "${loss[@]}"; do
	    	for m in "${multi[@]}"; do
	    		for c in "${c[@]}"; do
	    			for t in "${tol[@]}"; do
	    				for i in "${inter[@]}"; do
	    					logfile="${out_directory}/${iter}_results.txt"
	    					sqsub -f threaded -n $CPU --mpp=$MEMORY -r $WALL_TIME -o $logfile python classifier.py $c $l $t $e $m $i $datafile
							((iter++))
						done
					done
				done
			done
		done
	done
}



#Execution section
check_directory $out_directory
run_tests

