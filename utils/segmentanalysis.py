import Levenshtein as lv

def segmentphonelist(listofphones,grouplessthan=0,skip=0,printlevel=0):
    segments = [] # tracks segments
    """RNNs performs generally fairly bad on the first
    few classifications, adding option to skip"""
    cphone = listofphones[skip]
    count = 1
    cstart = skip+1
    for i,phone in enumerate(listofphones[cstart:]):
        if phone == cphone:
            count += 1
        else:
            cend = cstart+count
            segments.append((cphone,cstart,cend,count))
            count = 1
            cphone = phone
            cstart = cend
    cend = cstart + count-2
    segments.append((phone, cstart, cend, count))
    csegs = []
    for s in segments: # re. groupslessthan
        if s[3] > grouplessthan:
            csegs.append(s)
    if printlevel == 1:
        print(csegs)
    if printlevel == 2:
        print(f'\t No skip :{segments}')
        print(f'\t Cleaner :{csegs}')
    return csegs

def seg2numstring(seglist):
    return ''.join(c[0] for c in seglist)


def uttLD(gstd, syth):
    gstd_string = seg2numstring(gstd)
    syth_string = seg2numstring(syth)
    answer = lv.distance(gstd_string, syth_string)
    return answer

# Compare
def segCorrect(gstd,syth):
    return len([1 for i,j in zip(gstd,syth) if i == j])
