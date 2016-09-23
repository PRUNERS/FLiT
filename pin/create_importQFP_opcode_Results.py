CREATE OR REPLACE FUNCTION importOpcodeResults(path text, run integer) RETURNS integer[] as $$
import glob
from plpy import spiexceptions
import os
count = 0
skipped = 0
for f in glob.iglob(path + '/*'):
    fels = os.path.basename(f).split('_')
    if len(fels) != 5:
        continue
    if fels[0] == 'INTEL':
        compiler = 'icpc'
    elif fels[0] == 'GCC':
        compiler = 'g++'
    elif fels[0] == 'CLANG':
        compiler = 'clang++'
    dynamic = False
    host = fels[1]
    flags = fels[2]
    precision = fels[3]
    name = fels[4]
    tq = ("SELECT index from tests where " +
          "name = '" + name + "' and " +
          "host = '" + host + "' and " +
          "precision = '" + precision + "' and " +
          "compiler = '" + compiler + "' and " +
          "switches = (select switches from switch_conv where abbrev = '" + flags + "') and " +
          "run = " + str(run))
    res = plpy.execute(tq)
    if res.nrows() == 0:
        print('test record not found for: ' + tq)
        print('ignoring . . .')
        skipped = skipped + 1
        continue
    tindx = res[0]["index"]
    with open(f) as inf:
        for line in inf:
            l = line.split()
            if len(line.lstrip()) > 0 and line.lstrip()[0] == '#':
                if 'dynamic' in line:
                    dynamic = True
                continue
            if len(l) < 4:
                continue
            opq = ("INSERT INTO opcodes VALUES(" +
                   str(l[0]) + ", '" + l[1] +"')")
            try:
                plpy.execute(opq)
            except spiexceptions.UniqueViolation:
                pass

            cntq = ("INSERT INTO op_counts (test_id, opcode, " +
                    "count, pred_count, dynamic) "+
                    "VALUES(" + str(tindx) + ", " + str(l[0]) +
                    ", " + str(l[2]) + ", " + str(l[3]) + ", " + str(dynamic) + ")")
            plpy.execute(cntq)
            count = count + 1
return [count, skipped]
$$ LANGUAGE plpython3u;
   
