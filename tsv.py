import re
for l in open("xor_profile.txt").readlines():
    l = l.strip().rstrip("\r\n")
    l = re.sub(" +", "\t", l)
    print(l)
    
    
    
