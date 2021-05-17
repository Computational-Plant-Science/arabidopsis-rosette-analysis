"""
Version: 1.5

Summary: Plant image traits computation pipeline

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python pipeline.py -p /home/suxingliu/plant-image-analysis/random_test/ -ft jpg

parameter list:

 ap.add_argument("-p", "--path", required = True, help = "path to image file")
 ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")


"""

import subprocess, os
import sys
import argparse

'''
def execute_script(cmd_line):
    """execute script inside program"""
    try:
        print(cmd_line)
        
        process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE)
        
        process.wait()
        
        #print process.returncode
        
    except OSError:
        
        print("Failed ...!\n")

'''
def execute_script(cmd_line):
    """execute script inside program"""
    
    process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        nextline = str(process.stdout.readline())
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        print("failed!...\n")
        

def image_analysis_pipeline(file_path, ext):
    """execute pipeline scripts in order"""
    
    #python /opt/code/trait_extract_parallel.py -p /home/suxingliu/plant-image-analysis/data/ -ft JPG
    # step 1
    trait_extract_parallel = "python /opt/code/trait_extract_parallel.py -p " + file_path + " -ft " + str(ext) 
    
    print("Plant image traits computation pipeline...\n")
    
    execute_script(trait_extract_parallel)
       
    

if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    args = vars(ap.parse_args())
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']

   
    image_analysis_pipeline(file_path, ext)
    
    
