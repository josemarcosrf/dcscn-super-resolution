#!/bin/bash

say() {
 echo "$@" | sed \
         -e "s/\(\(@\(red\|green\|yellow\|blue\|magenta\|cyan\|white\|reset\|b\|u\)\)\+\)[[]\{2\}\(.*\)[]]\{2\}/\1\4@reset/g" \
         -e "s/@red/$(tput setaf 1)/g" \
         -e "s/@green/$(tput setaf 2)/g" \
         -e "s/@yellow/$(tput setaf 3)/g" \
         -e "s/@blue/$(tput setaf 4)/g" \
         -e "s/@magenta/$(tput setaf 5)/g" \
         -e "s/@cyan/$(tput setaf 6)/g" \
         -e "s/@white/$(tput setaf 7)/g" \
         -e "s/@reset/$(tput sgr0)/g" \
         -e "s/@b/$(tput bold)/g" \
         -e "s/@u/$(tput sgr 0 1)/g"
}

# check current directory
current_dir=${PWD##*/}
if [ "$current_dir" == "scripts" ]; then
    say @red[["This scripts should be executed from the root folder as: ./scripts/train_nlu.sh"]]
    exit
fi

export LASER=$PWD

set -Eeuo pipefail

function print_help {
    echo "Available options:"
    echo " train + (train cmdline arguments)            - Start training"
    echo " eval + (not implemented yet)                 - Start evaluation of a trained model"
    echo " train -h                                     - Print train help"
    echo " eval -h                                      - Print eval help"
    echo " help                                         - Print this help"
    echo " run                                          - Run an arbitrary command inside the container"
}

case ${1} in
    train)
        exec python train.py "${@:2}"
        ;;
    eval)
        exec python eval.py "${@:2}"
        ;;
    run)
        exec "${@:2}"
        ;;
    *)
        print_help
        ;;
esac
