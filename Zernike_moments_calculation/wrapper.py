from zernike_preprocessing_wrapper import main_pre
from zernike_postprocessing_wrapper import main_post
from glob import glob
import subprocess

f = glob("*.csv")
for i in f:
    main_pre(i)
    subprocess.run("./ZernikeMomentsC rad_poly.txt > moments.txt",shell = True,check = True)
    main_post(i[:-4],i)
