import subprocess, os
import sys

'''
    The code to run the workflow code on a sample
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
        print("Failed!")
        
        
def process_sample(name,path,args):
    '''
        Process a sample within the collection.

        This function is run within the singularity container defined in
        the WORKFLOW_CONFIG using the python3 interpreter.

        Args:
            name (str): name of the sample
            path (str): path to the sample file(s)
            args (dict): workflow parameters

        Returns:
            A python dictionary with any/all of the following keys:

            'key-val': a dictionary of key value pairs. These are accumulated
                from all the process_sample calls and placed in a csv file.
                Values must be primitives (float,int,string).

                Keys do not need to be the same across all samples. Keys missing
                from samples are set to NULL in the resulting csv file.

            'files': a list of paths to the files to include in the results.
                The files are placed in a "files" folder and compressed.

                The paths must be relative to the sample directory, which is the
                current directory when this script is called.

                Files names do not need to be unique between samples.
    '''
    
    settings = args['settings']['params']
    
    trait_extract_parallel = "python /opt/code/trait_extract_parallel.py -p " + current_path + "/" + " -ft " + str(settings['filetype'])
    
    print("Analyzing the traits of plant images...\n")
    
    execute_script(trait_extract_parallel)
    
    
    
    return {'files':[ current_path + '/results']}
