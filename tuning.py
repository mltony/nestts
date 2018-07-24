from datetime import datetime
import math
import multiprocessing
import numpy as np
import os
import util


class FunctionCaller(object):
    def __init__(self, kwargs):
        self.kwargs = dict(kwargs)
        self.seed_base= self.pop("seed_base", 0)
        self.func = self.pop("func")
        self.func_name = self.pop("func_name")
        self.path= self.pop("name", "")
        if len(self.path) > 0:
            try:
                os.mkdir(self.path)
            except FileExistsError:
                pass
            
    def pop(self, var, default=None):
        try:
            return self.kwargs.pop(var)
        except KeyError:
            if default is None:
                raise
            return default
        
        
    def __call__(self, seed):
        seed += self.seed_base
        try:
            result = self.func(
                **self.kwargs,
                seed=seed ,
                out_file_name=os.path.join(self.path, "s%02d-out.txt" % seed),
                orgs_file_name=os.path.join(self.path, "s%02d-orgs.txt" % seed)
                )
            print(self.func_name + "(%d) = %s" % (seed, result))
            return result
        except:
            #raise
            print("Error on seed=%d" % seed)
            return (math.nan, math.nan)  


    
    
def func_avg(kwargs, f=None, name=None):
    kwargs = dict(kwargs)
    pool = kwargs.pop("pool")
    n_runs = kwargs.pop("n_runs")
    result = pool.map(FunctionCaller(kwargs), range(n_runs))
    xs,sizes = list(zip(*result))
    if f is None:
        f = open("tuning.txt", "w") 
    xs_str = "Evals: " + str(xs)
    sizes_str = "Sizes: " + str(sizes)
    median_str = "Median: %.1f  Average: %.1f ~ %.1f" % (util.median(xs), util.avg(xs), util.avg([x for x in xs if util.is_finite(x)]))
    avg_size_str = "AverageSize: %.1f ~ %.1f" % (util.avg(sizes), util.avg([s for s in sizes if util.is_finite(s)]))
    success_rate = len([x for x in xs if util.is_finite(x)]) / len(xs) 
    success_rate_str = "Success_rate: %.2f" % success_rate
    report_str = [xs_str, sizes_str, median_str, avg_size_str,success_rate_str]
    if name is not None:
        name_str = "* %s" % name
        report_str = [name_str] + report_str
    report_str = "\n".join(report_str)
    print(report_str, file=f)
    f.flush()
    print(report_str)

def func_spread(kwargs, var, values, f=None, prefix=None): 
    if f is None:
        f = open("tuning.txt", "w")
    t1 = datetime.now()
    print("%s: Starting spread for %s" % (str(t1), var), file=f)
    f.flush()        
    kwargs = dict(kwargs)
    for value in values:
        kwargs[var] = value
        name = "%s_%s" % (var, value)
        if prefix is not None:
            name = "%s_%s" % (prefix, name)
        kwargs["name"] = name
        func_avg(kwargs, f=f, name=name)
    t2 = datetime.now()
    print("%s: Finished spread for %s, time=%s" % (str(t2), var,str(t2-t1)), file=f)
    f.flush()


