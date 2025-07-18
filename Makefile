GCC = g++
FILES = learn.cpp
FILES_MT = learn_mt.cpp

release: o2

release-multi: multi-o2

multi: release-multi

o1:
	$(GCC) -g -Wall -Wextra -O1 $(FILES) -o bin-learn-o1

o2:
	$(GCC) -g -Wall -Wextra -O2 $(FILES) -o bin-learn-o2

o3:
	$(GCC) -g -Wall -Wextra -O3 $(FILES) -o bin-learn-o3

multi-o1:
	$(GCC) -g -Wall -Wextra -O1 $(FILES_MT) -o bin-learn-multi-o1

multi-o2:
	$(GCC) -g -Wall -Wextra -O2 $(FILES_MT) -o bin-learn-multi-o2

multi-o3:
	$(GCC) -g -Wall -Wextra -O3 $(FILES_MT) -o bin-learn-multi-o3


all: o1 o2 o3 multi-o1 multi-o2 multi-o3

debug:
	$(GCC) $(FILES) -o bin-learn

