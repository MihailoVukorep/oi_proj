GCC = g++
FILES = learn.cpp

release: o2

o1:
	$(GCC) -g -Wall -Wextra -O1 $(FILES) -o bin-learn-o1

o2:
	$(GCC) -g -Wall -Wextra -O2 $(FILES) -o bin-learn-o2

o3:
	$(GCC) -g -Wall -Wextra -O3 $(FILES) -o bin-learn-o3

all: release o1 o2 o3

debug:
	$(GCC) $(FILES) -o bin-learn

