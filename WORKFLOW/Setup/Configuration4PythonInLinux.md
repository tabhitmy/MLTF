##Configuration of the installation of python in server with linux os. 

* Step 0. Make the connection of each node. And load them with root

* Step 1. Prepare the Installation files into the /usr/local/python36 folder. 

            cp -r /home/labcompute/GaoMY/PYTHONLINUX/* /usr/local/python36/

* Step 2. Unzip the Python-3.6.1.tgz

            cd /usr/local/python36/
            tar -zxvf Python-3.6.1.tgz

* step 3. Get into the ~/Python-3.6.1 
        run these lines of code: 

            ./configure -prefix=/usr/local/python36/Python-3.6.1
            make 
            rpm -i zlib(with version name).rpm
            (install zlib, it lies in ~/python36)
            make install 

* step 4. Line the python and pip with command 
        run these two lines of code: 
    
        ln -s /usr/local/python36/Python-3.6.1/bin/python3.6 /usr/local/bin/python3

       
        ln -s /usr/local/python36/Python-3.6.1/bin/pip3.6 /usr/local/bin/pip

----

Till now , the basic python is installed. Then continue with the intel MKL.

-------


* step 5.  Installation of MKL 

    1. Change directory back to ~/python36:
        cd /usr/local/python36
    2. Unzip the MKL:
        tar -zxvf l_MKL(with version info).gz
    3. run the shell then follow the installation wizard to finish the installation of the MKL :
        ./install.sh\v

* step6. Add MKL lib into the path 

    1.run:
        vi /etc/profile 
    2.press 'i' and Add two line at the end: 
        export PATH=":/opt/intel/bin:$PATH"
        export LD_LIBRARY_PATH=$LD_LIBRARY_LIB:/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64
    3.press 'esc', then press ':',then type'wq', then press 'enter' to save
    4.run:
        source /etc/profile

* step7. Install the MKL_python distribution 
        same with step 5

* step8. Install this list of python packages with this order

		pip install numpy-1.13.0-cp36-cp36m-manylinux1_x86_64.whl &
		pip install pyparsing-2.2.0-py2.py3-none-any.whl &
		pip install six-1.10.0-py2.py3-none-any.whl &
		pip install python_dateutil-2.6.0-py2.py3-none-any.whl &
		pip install cycler-0.10.0-py2.py3-none-any.whl &
		pip install pytz-2017.2-py2.py3-none-any.whl &
		pip install matplotlib-2.0.2-cp36-cp36m-manylinux1_x86_64.whl &
		pip install scikit_learn-0.18.1-cp36-cp36m-manylinux1_x86_64.whl &
		pip install pandas-0.20.2-cp36-cp36m-manylinux1_x86_64.whl &
		pip install scipy-0.19.0-cp36-cp36m-manylinux1_x86_64.whl &
		pip install patsy-0.4.1-py2.py3-none-any.whl &
		pip install statsmodels-0.8.0-cp36-cp36m-manylinux1_x86_64.whl &


* step9. Checklis for Python package

	    cycler (0.10.0)
	    matplotlib (2.0.2)
	    numpy (1.13.0)
	    pandas (0.20.2)
	    patsy (0.4.1)
	    pip (9.0.1)
	    pyparsing (2.2.0)
	    python-dateutil (2.6.0)
	    pytz (2017.2)
	    scikit-learn (0.18.1)
	    scipy (0.19.0)
	    setuptools (28.8.0)
	    six (1.10.0)
	    statsmodels (0.8.0)


* Addtional step 


	1. Installation of line_profiler. 
		PACKAGES COLLECTION(Quite a lot)

	2. Add to path 
		ln -s /usr/local/python36/Python-3.6.1/bin/kernprof /usr/local/bin/kernprof	
	
	3. Execution 
		kernprof -l -v test.py


* Additonal step 
	1. Installation of the keras 
		PyYmal 
		theano 



* when the Module got problem 


export PYTHONPATH=$PYTHONPATH:"/usr/local/python36/Python3.6.1/lib/python3.6/site-packages"



* When encountering a problem with module '_bz2' . refer these websites:  
         https://lists.gt.net/python/python/1029957
         https://my.oschina.net/u/183476/blog/1212739
         http://www.111cn.net/sys/CentOS/62830.htm
         * Details: 
             1. install the bzip2-1.0.5   
                        unzip the file
                        go to the folder
                        $make 
                        $make -f Makefile-libbz2_so (make it working. if not, please do the  'rm -vf /usr/bin/bz* ')
                        $make install 
            2.  Go to the python path 
                        $ ./configure --with-bz2=/usr/local/include 
                        $ LB_LIBRARY_PATH=/usr/local/lib python3 -c 'import bz2'
                        $ make 
                        $ make install 


* Updating GLIBC 
xz -d glibc-2.17.tar.xz
tar -xvf glibc-2.17.tar
cd glibc-2.17
mkdir build
cd build
../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin  
make && make install
需要等大概10分钟。

输入strings /lib64/libc.so.6|grep GLIBC发现已经更新 



* Updating GLIBCXX

从网上下载libstdc++.so.6.0.20 

http://ftp.de.debian.org/debian/pool/main/g/gcc-4.8/
或者
http://download.csdn.net/detail/pomelover/7524227


放到/usr/lib64/下
#chmod +x libstdc++.so.6.0.20
#rm libstdc++.so.6
#ln -s libstdc++.so.6.0.20 libstdc++.so.6
#strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX

                    