bench:
	echo "======= building pw2-omp ======="
	cd ./pw2-omp && make clean && make
	echo "======= building pw2-cuda ======="
	cd ./pw2-cuda && make clean && make
	echo "======= START benchmark ======="
	/home/hy11254z/miniforge3/envs/py311/bin/python benchmark.py	

clean-omp:
	rm csvs/ex1-*.csv

clean-cuda:
	rm csvs/ex2-*.csv