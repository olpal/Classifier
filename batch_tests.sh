ROOT_DIR=/scratch/aolpin/project
MEMORY=6G
CPU=4
WALL_TIME=4h
log_options=( 2 3 4 5 )
avp_options=( 4 5 7 )
sqr_options=( 2 3 6 )
norm_options=( 6 7 )
alpha=( 1 10 50 100 500 1000 )
max_terms=(5 10 20 50 80 100)
option_names=( normal logsquareadd logsquaremult logavpadd logavpmult scoresquare scoreavp)
test_name="test-1-hyper"
out_directory="${ROOT_DIR}/${test_name}"
max_iter=1

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

#Runs Normal tests
function run_normal_tests(){
	check_directory "${out_directory}"
	counter=1
	while [ $counter -le $max_iter ]; do
		option_name=${option_names[1]}
	    #echo "$counter"
		logfile="${out_directory}/${test_name}-${counter}-${option_name}-results.txt"
	    sqsub -f threaded -n $CPU --mpp=$MEMORY -r $WALL_TIME -o $logfile python classifier.py 1				
	((counter++))
	done
}

function run_log_avp_tests(){
	check_directory "${out_directory}"
	counter=1
	while [ $counter -le $max_iter ]; do
		for o in "${avp_options[@]}"; do
			option_name=${option_names[(($o-1))]}
			logfile="${out_directory}/${test_name}-${counter}-${option_name}-results.txt"
	    	sqsub -f threaded -n $CPU --mpp=$MEMORY -r $WALL_TIME -o $logfile python classifier.py $o 50 20
		done				
	((counter++))
	done
}

function run_log_sqr_tests(){
	check_directory "${out_directory}"
	counter=1
	while [ $counter -le $max_iter ]; do
		for o in "${sqr_options[@]}"; do
			option_name=${option_names[(($o-1))]}
			logfile="${out_directory}/${test_name}-${counter}-${option_name}-results.txt"
	    	sqsub -f threaded -n $CPU --mpp=$MEMORY -r $WALL_TIME -o $logfile python classifier.py $o 10 20
		done				
	((counter++))
	done
}

function run_log_tests_hyper(){
	check_directory "${out_directory}"
	counter=1
	while [ $counter -le $max_iter ]; do
		for o in "${log_options[@]}"; do
			option_name=${option_names[(($o-1))]}
			logfile="${out_directory}/${test_name}-${counter}-${option_name}-results.txt"
			for a in "${alpha[@]}"; do
				for m in "${max_terms[@]}"; do
	    			sqsub -f threaded -n $CPU --mpp=$MEMORY -r $WALL_TIME -o $logfile python classifier.py $o $a $m
	    		done
	    	done
		done				
	((counter++))
	done
}

run_normal_tests
run_log_avp_tests
run_log_sqr_tests