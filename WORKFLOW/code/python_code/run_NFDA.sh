#!/bin/bash
clear

username="GaoMY" 
computeno=95 # please use 93 or 95 or 96

pathprefix="/EXECUTION/NFDA/code"
serverpath="~/$username$pathprefix"
localpath="../"

ssh -l labcompute 192.168.0.$computeno "export PYTHONPATH=$PYTHONPATH:\"/usr/local/python36/Python3.6.1/lib/python3.6/site-packages\"; export USERNAME=$username; python3 ~/public/backend_data/NFDAkey.py"
passflag=$(ssh -l labcompute 192.168.0.$computeno "echo $?")

if [ $passflag = 0 ];then
    exit
fi


sleep 2
# Create the folder in slave computer  [ ../code]
ssh -l labcompute 192.168.0.$computeno "mkdir -p $serverpath/python_code/; rm $serverpath/python_code/*.py" 
wait 
ssh -l labcompute 192.168.0.$computeno "mkdir $serverpath/python_code/fig/; rm $serverpath/python_code/fig/*.png "
wait 
ssh -l labcompute 192.168.0.$computeno "mkdir $serverpath/python_code/data/"
wait
ssh -l labcompute 192.168.0.$computeno "rm -rf  $serverpath/tlog.txt"
wait 

if [ "$username" = "GaoMY" ]; then
    echo "########################################"
    echo "#[Update to the repository. Wlkm Commander!] #"
    echo "########################################"
    rm -rf /home/public/GaoMY/EXECUTION/NFDA/code/python_code/*  
    cp -r $localpath/python_code/*.py /home/public/GaoMY/EXECUTION/NFDA/code/python_code &
    cp -r $localpath/python_code/*.sh /home/public/GaoMY/EXECUTION/NFDA/code/python_code
    wait
    echo "########################################"
    echo "#[Deliver passcode to all slaves] #"
    echo "########################################"
    scp -r $localpath/NFDAkey.py labcompute@192.168.0.71:~/public/backend_data/ &
    scp -r $localpath/NFDAkey.py labcompute@192.168.0.95:~/public/backend_data/ &
    scp -r $localpath/NFDAkey.py labcompute@192.168.0.96:~/public/backend_data/ &
    scp -r $localpath/NFDAkey.py labcompute@192.168.0.93:~/public/backend_data/ &
    wait
    echo "########################################"
    echo "#[Back up onto slave 2, 192.168.0.71] #"
    echo "########################################"
    ssh -l labcompute 192.168.0.71 "rm -rf $serverpath/python_code/*; mkdir -p $serverpath/python_code/; "&
    scp -r $localpath/python_code/*.py labcompute@192.168.0.71:$serverpath/python_code &
    scp -r $localpath/python_code/*.sh labcompute@192.168.0.71:$serverpath/python_code & 
    # Update the data to slave
    scp -r  $localpath/python_code/data labcompute@192.168.0.$computeno:$serverpath/python_code/ &
fi

wait 
echo "########################################"
echo "######## [Updating the Code...] ########"
echo "########################################"

scp -r  $localpath/python_code/*.py labcompute@192.168.0.$computeno:$serverpath/python_code &
wait

echo "########################################"
echo "######### [ Computing ... ] #########"
echo "########################################"
echo "Please check in ../tlog.txt for the log"

ssh -l labcompute 192.168.0.$computeno "export PYTHONPATH=$PYTHONPATH:\"/usr/local/python36/Python3.6.1/lib/python3.6/site-packages\"; export USERNAME=$username; python3 $serverpath/python_code/NFDALauncher.py"
wait
echo "########################################"
echo "######### [Done! Fetching ...] #########"
echo "########################################"
scp -r labcompute@192.168.0.$computeno:$serverpath/python_code/fig $localpath/python_code/fig
scp -r labcompute@192.168.0.$computeno:$serverpath/tlog.txt $localpath/



