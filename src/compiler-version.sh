[[ `gcc --version` =~ ([0-9]).([0-9]).([0-9]) ]]

if [ ${BASH_REMATCH[1]} -eq 4 ]
then
	if [ ${BASH_REMATCH[2]} -eq 4 ]
	then
		echo OK
		exit 0
	fi
fi

if [ -f `which gcc-4.4` ]
then
	mkdir -p compiler-bindir
	ln -sf `which gcc-4.4` compiler-bindir/gcc
	ln -sf `which g++-4.4` compiler-bindir/g++
	echo FIXED
	exit 1
fi

echo ERROR
exit 2
