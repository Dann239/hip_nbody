#!/bin/bash
files=""
platform="guess"
name="a.out"
run=0
while [ -n "$1" ]
do
case "$1" in
-r) run=1 ;;
-p) platform="$2" 
shift ;;
-o) name="$2"
shift ;;
*) files="$files $1" ;;
esac
shift
done
#echo $platform
if [ $platform = "nvcc" ]
then
nvcc $files -o $name
fi
if [ $platform = "hip" ]
then
    hiped=""
    filesnew=""
    for currfile in $files
    do
        #echo $currfile
        if grep -q ".cu" <<< "$currfile"; then
            currfilenew=${currfile//".cu"/}
            currfilenew="$currfilenew.cpp"
            hipify-clang --extra-arg="-std=c++14" -o=$currfilenew $currfile
            hiped="$hiped $currfilenew"
        else
            filesnew="$filesnew $currfile"
        fi
    done
    filesnew="$filesnew $hiped"
    hipcc $filesnew -std=c++14 -o $name
    #echo $filesnew
    #echo $hiped
    for currfile in $hiped
    do
        #echo $currfile
        rm $currfile
    done
fi

if [ $run -eq 1 ] 
then
./$name
fi

