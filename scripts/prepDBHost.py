#!/usr/bin/env python3

#this script prepares the db host by making a place to
#push the output files to so that the import processes may
#be executed upon them

COLL_DIR = 'flit_data'

if __name__ == '__main__':

    import tempfile, os, glob
    from subprocess import check_output
    import shutil

    TAR = check_output('which tar', shell=True, universal_newlines=True)[:-1]

    try:
        os.mkdir(COLL_DIR)
    except FileExistsError:
        if len(glob.glob(COLL_DIR + '/*')) > 0:
            os.chdir(COLL_DIR)
            f = tempfile.mkstemp('.tgz', dir='..', prefix=COLL_DIR + '_')
            stdot = check_output(TAR + ' zcf ' + f[1] + ' *', shell=True,
                                 universal_newlines=True)
            os.chdir('..')
            shutil.rmtree(COLL_DIR)
            os.mkdir(COLL_DIR)
            print('created ' + f[1])
        else:
            print(COLL_DIR + ' was empty')
            exit(0)
    print('made directory ' + COLL_DIR)
    exit(0)
    
