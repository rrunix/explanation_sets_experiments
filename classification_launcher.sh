# Kill everything when the script ends..
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Split by cores (weird hyperopt behaviour that forks several processes with max usage with no perf. gain...)
taskset --cpu-list 0 python classification_experiments.py 0 291 &> /dev/null &
taskset --cpu-list 1 python classification_experiments.py 291 583 &> /dev/null &
taskset --cpu-list 2 python classification_experiments.py 583 875 &> /dev/null &
taskset --cpu-list 3 python classification_experiments.py 875 1166 &> /dev/null &
taskset --cpu-list 4 python classification_experiments.py 1166 1458 &> /dev/null &
taskset --cpu-list 5 python classification_experiments.py 1458 1750 &> /dev/null &
taskset --cpu-list 6 python classification_experiments.py 1750 2041 &> /dev/null &
taskset --cpu-list 7 python classification_experiments.py 2041 2333 &> /dev/null &
taskset --cpu-list 8 python classification_experiments.py 2333 2625 &> /dev/null &
taskset --cpu-list 9 python classification_experiments.py 2625 2917


# When the last script ends.. run through all explanations
taskset --cpu-list 9 python classification_experiments.py
