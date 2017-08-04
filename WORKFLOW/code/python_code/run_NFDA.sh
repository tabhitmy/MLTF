#!/bin/bash
clear
pathprefix="/EXECUTION/NFDA/code"
username="GaoMY"   
serverpath="~/$username$pathprefix"
localpath="../"


ssh -l labcompute 192.168.0.95 "rm -rf $serverpath/python_code/; mkdir $serverpath/python_code/"& 
wait
echo "########################################"
echo "######## [Updating the Code...] ########"
echo "########################################"
# scp -r  $localpath/data/ labcompute@192.168.0.95:$serverpath/ &
scp -r  $localpath/python_code/ labcompute@192.168.0.95:$serverpath/ &
wait

echo "########################################"
echo "######### [ Initializing ... ] #########"
echo "########################################"
ssh -l labcompute 192.168.0.95 "export PYTHONPATH=$PYTHONPATH:\"/usr/local/python36/Python3.6.1/lib/python3.6/site-packages\"; python3 $serverpath/python_code/NFDALauncher.py"
