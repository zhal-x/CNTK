import argparse
import importlib
import os
import platform
import re
import signal
import subprocess
import sys
import time

num_logs_to_collect = 10
max_per_test_time = 300 # in seconds

def load_test_cases(name):
    if ('.' in name):
        i = name.find('.')
        file = name[0:i]
        test_pattern = name[i+1:]
        tests = importlib.import_module(file).test_cases
        if ('*' in test_pattern or '?' in test_pattern):
            test_pattern = test_pattern.replace('*', '.*') # to get regex work with wildcard *
            test_pattern = test_pattern.replace('?', '.') # to get regex work with wildcard ?
            out = {}
            for n in tests:
                if (len(re.findall(test_pattern, n)) > 0):
                    out[n] = tests[n]
            return out
        else:
            return {test_pattern:tests[test_pattern]}
    else:
        return importlib.import_module(name).test_cases
        
def parse_cntkv1_output(line):
    results = re.findall("c. = ([^ ]+?) \* (.+?); errs = (.+?)% .*; .*; epochTime=(.+?)s", line)
    return [float(v) for v in list(results[0])] if len(results) > 0 else [] # ce, samples, err, seconds

def parse_cntkv2_output(line):
    '''Finished Epoch[7 of 10]: loss = 0.042747 * 60032, metric = 0.0% * 60032 1.219s (49246.9 samples per second)'''
    results = re.findall("loss = (.+?) \* (.+?), metric = (.+?)% \* [0-9]+ (.+?)s ", line)
    return [float(v) for v in list(results[0])] if len(results) > 0 else [] # ce, samples, err, seconds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance test')
    parser.add_argument('-t', '--run_test', help='<test case file>[.test_case_name|pattern]', required=True, type=str)
    args = vars(parser.parse_args())
    testconfig = args['run_test']

    test_cases = load_test_cases(testconfig)
    sorted_key = sorted(test_cases)
    
    cwd = os.getcwd()
    verbose = (len(sorted_key) == 1)
    
    if not verbose:
        print("Selected tests: {}".format(sorted_key))
    
    for n in sorted_key:
        print("\n--- Running  {} ---\n".format(n))
        t = test_cases[n]
        start_time = time.time()
        path = ["..", "..", ".."]+t["dir"].split("/")
        os.chdir(os.path.join(*path))
        training_samples = 0
        training_seconds = 0
        try:
            args = t["args"]
            is_cntkV2 = (t["exe"] == "python.exe")
            
            # add default args
            if is_cntkV2:
                args = ["-u"] + args # this is needed for python to output without caching
            else:
                args = args + ["makeMode=false", "traceLevel=1"] # need to be verbose at the beginning to fill out stdout before capturing lines
            
            args = [t["exe"]] + args
            
            distributed = ("distributed" in t)
            
            if distributed:
                # add tags of node
                if platform.system() == 'Linux':
                    fixture = ["-tag-output"]
                else:
                    fixture = ["-lines"]
                args = ["mpiexec"] + fixture + t["distributed"] + args

            with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                logs = 0
                for line in p.stdout:
                    if verbose:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    
                    #preprocess lines to parse only rank 0
                    if distributed:
                        if platform.system() == 'Linux':
                            line = line[3:]
                        else:
                            if line[0:3] == "[0]":
                                line = line[3:]
                            else:
                                line = ""
                    
                    result = parse_cntkv2_output(line) if is_cntkV2 else parse_cntkv1_output(line)
                    if (len(result) > 0):
                        if not verbose:
                            print("ce {} err {} speed {:.1f}".format(result[0], result[2], result[1]/result[3]))
                        training_samples += result[1]
                        training_seconds += result[3]
                        logs+=1

                    #check if we"ve got enough log, or if the process has run long enough
                    if (logs > num_logs_to_collect or time.time() - start_time > max_per_test_time):
                        os.kill(p.pid, signal.CTRL_C_EVENT)
        except KeyboardInterrupt:
            pass
        os.chdir(cwd)
        print("\n--- Finished {}: {:.0f} samples in {:.1f} seconds ({:.1f} samples/second), total {:.1f} seconds ---\n".format(
            n, training_samples, training_seconds, training_samples/training_seconds if training_seconds>0 else 0, time.time() - start_time))
