#!/usr/bin/env python3

import sys
import re
import numpy as np

def extract(starfile):
    
    header_found = False
    with open(starfile, "r") as openfile:
        for line in openfile:
            if re.search(r'loop_', line):
                header_found = True
                continue # skip to next iteration of loop
                
            if header_found == True and line[0] == '_':
                if re.search(r'_rlnImageSize', line):
                    idx_box = int(str(line.split()[1]).strip("#"))
                    
                if re.search(r'_rlnImagePixelSize', line):
                    idx_apix_box = int(str(line.split()[1]).strip("#"))
                    
                if re.search(r'_rlnMicrographOriginalPixelSize', line):
                    idx_apix_img = int(str(line.split()[1]).strip("#"))
            
            elif header_found == True and line[0] != '_':
                box = (line.split()[idx_box-1])
                apix_box = (line.split()[idx_apix_box-1])
                apix_img = (line.split()[idx_apix_img-1])
                break
                
    meta = [box, apix_box, apix_img]
    return meta