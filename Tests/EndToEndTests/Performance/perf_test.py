import importlib
import time
import subprocess
import argparse
import os
import signal
import re

num_logs_to_collect = 10
max_per_test_time = 300 # in seconds

def load_test_cases(name):
    if ('.' in name):
        i = name.find('.')
        file = name[0:i]
        test_pattern = name[i+1:]
        test_pattern = test_pattern.replace('*', '.*') # to get regex work with wildcard *
        test_pattern = test_pattern.replace('?', '.') # to get regex work with wildcard ?
        tests = importlib.import_module(file).test_cases
        out = {}
        for n in tests:
            if (len(re.findall(test_pattern, n)) > 0):
                out[n] = tests[n]
        return out
    else:
        return importlib.import_module(name).test_cases
        
def parse_cntkv1_output(line):
    results = re.findall("c. = ([^ ]+?) \* (.+?); errs = (.+?)% .*; .*; epochTime=(.+?)s", line)
    if (len(results) > 0):
        results = list(results[0])
        ce = results[0]
        err = results[2]
        speed = "{:.1f}".format(float(results[1])/float(results[3]))
        return [ce,err,speed]
    else:
        return []

def parse_cntkv2_output(line):
    '''Finished Epoch[7 of 10]: loss = 0.042747 * 60032, metric = 0.0% * 60032 1.219s (49246.9 samples per second)'''
    results = re.findall("loss = (.+?) .* metric = (.+?)% .* \((.+?) samples per second\)$$", line)
    return list(results[0]) if len(results) > 0 else [];

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance test')
    parser.add_argument('-t', '--run_test', help='<test case file>[.test_case_name]', required=True, type=str)
    args = vars(parser.parse_args())
    testcase = args['run_test']

    tests = load_test_cases(testcase)
    cwd = os.getcwd()
    
    for n in sorted(tests):
        print("\n--- Running {} ---\n".format(n))
        t = tests[n]
        start_time = time.time()
        path = ["..", "..", ".."]+t["dir"].split("/")
        os.chdir(os.path.join(*path))
        try:
            args = [*t["args"]]
            is_cntkV2 = (t["exe"] == "python.exe")
            
            # add default args
            if is_cntkV2:
                args = ["-u"] + args # this is needed for python to output without caching
            else:
                args = args + ["makeMode=false", "traceLevel=1"] # need to be verbose at the beginning to fill out stdout before capturing lines

            with subprocess.Popen([t["exe"]]+args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                logs = 0
                for line in p.stdout:
                    #print(line)
                    result = parse_cntkv2_output(line) if is_cntkV2 else parse_cntkv1_output(line)
                    if (len(result) > 0):
                        print("ce {} err {} speed {}".format(result[0], result[1], result[2]))
                        logs+=1

                    #check if we"ve got enough log, or if the process has run long enough
                    if (logs > num_logs_to_collect or time.time() - start_time > max_per_test_time):
                        os.kill(p.pid, signal.CTRL_C_EVENT)
        except KeyboardInterrupt:
            pass
        os.chdir(cwd)
        print("\n--- {}: {} seconds ---\n".format(n, time.time() - start_time))
