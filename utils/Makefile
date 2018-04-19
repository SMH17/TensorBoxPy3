SHELL := /bin/bash

.PHONY: all
all:
	pip3 install runcython3
	makecython3++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

hungarian: hungarian/hungarian.so

hungarian/hungarian.so:
	cd hungarian && \
	TF_INC=$$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
	TF_CFLAGS=( $$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') ) && \
	TF_LFLAGS=( $$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') ) && \
	if [ `uname` == Darwin ];\
	then g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared hungarian.cc -o hungarian.so -fPIC -I$$TF_INC/external/nsync/public/ -undefined dynamic_lookup -O3;\
	else g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared hungarian.cc -o hungarian.so -fPIC $${TF_CFLAGS[@]} $${TF_LFLAGS[@]} $$TF_INC -O3; fi
